import numpy as np, pandas as pd
import patsy
from drdid import reg_did
from csdid.attgt_fnc import drdid_trim

from csdid.utils.bmisc import panel2cs2
import warnings


fml = patsy.dmatrices
# Initialize a list to store data for each iteration
results_list = []

def compute_att_gt(dp, est_method = "dr", base_period = 'varying'):
    yname = dp['yname']
    tname = dp['tname']
    idname = dp['idname']
    xformla = dp['xformla']
    data = dp['data'].copy()
    weights_name = dp['weights_name']
    panel = dp['panel']
    true_rep_cross_section = dp['true_rep_cross_section']
    control_group = dp['control_group']
    anticipation = dp['anticipation']
    gname = dp['gname']
    n = dp['n']
    nT = dp['nT']
    nG = dp['nG']
    tlist = dp['tlist']
    glist = dp['glist']

    # Calculate time periods and adjustment factor
    tlist_len = len(tlist) - 1 if base_period != "universal" else len(tlist)
    tfac = 1 if base_period != "universal" else 0

    inf_func = []

    att_est, group, year, post_array = [], [], [], []

    def build_covariates(formula, frame):
        try:
            _, cov = fml(formula, data=frame, return_type='dataframe')
        except Exception as e:
            try:
                cov = patsy.dmatrix(formula, data=frame, return_type='dataframe')
            except Exception as e2:
                warnings.warn(f"Formula processing failed: {e2}")
                y_str, x_str = formula.split("~")
                xs1 = x_str.split('+')
                xs1_col_names = [x.strip() for x in xs1 if x.strip() != '1']
                n_dis = len(frame)
                ones = np.ones((n_dis, 1))
                try:
                    cov = frame[xs1_col_names].to_numpy()
                    cov = np.append(cov, ones, axis=1)
                except Exception:
                    cov = ones
        return np.array(cov)

    def add_att_data(att = 0, pst = 0, inf_f = []):
        inf_func.append(inf_f)
        att_est.append(att)
        group.append(g)
        year.append(tn)
        post_array.append(pst)

    # Handle never treated case
    never_treated = control_group == 'nevertreated'
    if never_treated:
        data['C'] = (data[gname] == 0).astype(int)
    data['y_main'] = data[yname]

    # Loop over groups
    for g_index, g in enumerate(glist):
        # Create a binary column 'G_m' to indicate if a row belongs to the current group 'g'
        G_main = (data[gname] == glist[g_index])
        data = data.assign(G_m=1 * G_main)

        # Loop over time periods
        for t_i in range(tlist_len):

            # Set pretreatment period
            pret = t_i  # Initialize pretreatment period as current time period index
            tn = tlist[t_i + tfac]  # Current time period (adjusted for tfac)

            # Universal base period
            if base_period == 'universal':
                try:
                    pret = np.where((tlist + anticipation) < g)[0][-1]
                except IndexError:
                    raise ValueError(
                        f"There are no pre-treatment periods for the group first treated at {g}. Units from this group are dropped."
                    )

            # For non-never treated groups, set up the control group indicator 'C'
            if not never_treated:
                n1 = (data[gname] == 0)
                n2 = (data[gname] > (tlist[max(t_i, pret) + tfac] + anticipation))
                n3 = (data[gname] != glist[g_index])
                row_eval = n1 | (n2 & n3)
                data = data.assign(C=1 * row_eval)

            # Check if in post-treatment period
            if glist[g_index] <= tlist[t_i + tfac]:
                pret_mask = (np.array(tlist) + anticipation) < glist[g_index]
                if not any(pret_mask):
                    warnings.warn(f"There are no pre-treatment periods for the group first treated at {glist[g_index]}\nUnits from this group are dropped")
                    break
                pret = np.where(pret_mask)[0][-1]

            pret_year = tlist[pret]
            post_treat = 1 * (g <= tn)
            if base_period == 'universal' and pret_year == tn:
                add_att_data(att=0, pst=post_treat, inf_f=np.zeros(n))
                continue

            # Subset the data for the current and pretreatment periods
            disdat = data[(data[tname] == tn) | (data[tname] == tlist[pret])]

        # results for the case with panel data
        #-----------------------------------------------------------------------------

            if panel:
                disdat = panel2cs2(disdat, yname, idname, tname)
                disdat = disdat.dropna()
                n = len(disdat)
                dis_idx = np.array(disdat.G_m == 1) | np.array(disdat.C == 1)
                disdat = disdat.loc[dis_idx, :]
                n1 = len(disdat)
                G = disdat.G_m
                C = disdat.C
                w = disdat.w

                ypre = disdat.y0 if tn > pret_year else disdat.y1
                ypost = disdat.y0 if tn < pret_year else disdat.y1
                covariates = build_covariates(xformla, disdat)

                G, C, w, ypre = map(np.array, [G, C, w, ypre])
                ypost, covariates = map(np.array, [ypost, covariates])

                if callable(est_method):
                    est_att_f = est_method
                elif est_method == "reg":
                    est_att_f = reg_did.reg_did_panel
                elif est_method == "ipw":
                    est_att_f = drdid_trim.std_ipw_did_panel
                elif est_method == "dr":
                    est_att_f = drdid_trim.drdid_panel

                att_gt, att_inf_func = est_att_f(ypost, ypre, G, i_weights=w, covariates=covariates)

                inf_zeros = np.zeros(n)
                att_inf = n / n1 * att_inf_func
                inf_zeros[dis_idx] = att_inf

                add_att_data(att_gt, pst=post_treat, inf_f=inf_zeros)

        #-----------------------------------------------------------------------------
        # results for the case with no panel data
        #-----------------------------------------------------------------------------

            if not panel:
                right_ids = disdat.loc[disdat.G_m.eq(1) | disdat.C.eq(1), 'rowid'].to_numpy()
                dis_idx = (data['rowid'].isin(right_ids)) & \
                            (data[tname].isin([tlist[t_i + tfac], tlist[pret]]))

                disdat = data.loc[dis_idx]

                G = disdat.G_m.to_numpy()
                C = disdat.C.to_numpy()
                Y = disdat[yname].to_numpy()
                post = 1 * (disdat[tname] == tlist[t_i + tfac]).to_numpy()
                w = disdat.w.to_numpy()
                n1 = sum(G + C)

                skip_this_att_gt = False

                if np.sum(G * post) == 0:
                    warnings.warn(f"No units in group {g} in time period {t_i + tfac + 1}")
                    skip_this_att_gt = True

                if np.sum(G * (1 - post)) == 0:
                    warnings.warn(f"No units in group {g} in time period {t_i + 1}")
                    skip_this_att_gt = True

                if np.sum(C * post) == 0:
                    warnings.warn(f"No available control units for group {g} in time period {t_i + tfac + 1}")
                    skip_this_att_gt = True

                if np.sum(C * (1 - post)) == 0:
                    warnings.warn(f"No available control units for group {g} in time period {t_i + 1}")
                    skip_this_att_gt = True

                if skip_this_att_gt:
                    add_att_data(att=np.nan, pst=post_treat, inf_f=np.full(n, np.nan))
                    continue

                covariates = build_covariates(xformla, disdat)

                if callable(est_method):
                    est_att_f = est_method
                elif est_method == "reg":
                    est_att_f = reg_did.reg_did_rc
                elif est_method == "ipw":
                    est_att_f = drdid_trim.std_ipw_did_rc
                elif est_method == "dr":
                    est_att_f = drdid_trim.drdid_rc

                att_gt, att_inf_func = est_att_f(y=Y, post=post, D = G, i_weights=w, covariates=covariates)
                att_inf_func = (n/n1)*att_inf_func

                inf_func_df = pd.DataFrame(
                {
                    "inf_func": att_inf_func,
                    "right_ids": right_ids
                }
                ).fillna(0)

                inf_zeros = np.zeros(n)
                aggte_infffuc = inf_func_df.groupby('right_ids').inf_func.sum()
                try:
                    dis_idx1 = np.isin(data['rowid'].unique(), aggte_infffuc.index.to_numpy())
                except Exception:
                    dis_idx1 = np.isin(data['rowid'].unique().to_numpy(), aggte_infffuc.index.to_numpy())

                inf_zeros[dis_idx1] = np.array(aggte_infffuc)

                add_att_data(att_gt, pst = post_treat, inf_f=inf_zeros)

    output = {
    'group': group,
    'year': year,
    'att': att_est,
    'post': post_array
    }

    return (output, np.vstack(inf_func))
