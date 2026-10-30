import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "mini_research_paper.tex"

J = lambda f: json.load(open(RES / f))
metrics, summary, bp = J("metrics.json"), J("data_summary.json"), J("best_params.json")
T, V = metrics["test"], metrics["validation"]
th, best, tune = bp["thresholds"], bp["best_params"], bp["tuning"]
root = summary["root_split_gini_example"]
depth_df = pd.read_csv(RES / "depth_diagnosis.csv")
lc = pd.read_csv(RES / "learning_curve.csv")
leaves = pd.read_csv(RES / "leaf_rules.csv")
sub = pd.read_csv(RES / "subgroup_metrics.csv")
imp = pd.read_csv(RES / "feature_importance.csv").set_index("feature")
ds = pd.read_csv(RES / "dataset_summary.csv")
cal = pd.read_csv(RES / "calibration_validation.csv")

def esc(s):
    return (str(s).replace("\\", "/").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            .replace("#", r"\#").replace("<=", r"$\leq$").replace(">", r"$>$").replace("<", r"$<$"))
tt = lambda s: r"\texttt{" + str(s).replace("_", r"\_") + "}"
f3 = lambda x: "undef." if x is None else f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
pc = lambda x, d=1: f"{100 * x:.{d}f}\\%"
nf = lambda x: f"{int(x):,}".replace(",", "{,}")
FEAT = "weekly_self_study_hours"

def table(caption, label, header, rows, spec, small=True):
    h = " & ".join(header) + r" \\"
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    sz = r"\small" if small else ""
    return (f"\\begin{{table}}[h!]\n\\centering\n{sz}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{h}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")

def figure(file, caption, label, width="0.9"):
    return (f"\\begin{{figure}}[h!]\n\\centering\n\\includegraphics[width={width}\\textwidth]{{{file}}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")

def row_metrics(name, e):
    return [name, f"{e['threshold']:.2f}", f3(e["accuracy"]), f3(e["balanced_accuracy"]), f3(e["precision"]),
            f3(e["recall"]), f3(e["specificity"]), f3(e["f1"])]

tf, tc, lf, lcst, mj = T["tree_F1_threshold"], T["tree_cost_threshold"], T["logistic_regression_F1_threshold"], T["logistic_regression_cost_threshold"], T["majority_class"]
costs = metrics["cost_per_student_test"]
n_fail = round(summary["fail_rate"] * summary["n_rows"])

t_desc = table("Descriptive statistics of the numerical variables ($n = " + nf(summary["n_rows"]) + "$).", "tab:desc",
    ["Variable", "Min", "Max", "Mean", "SD", "Median"],
    [[tt(r.variable) if r.variable != "total_score" else tt("total_score") + " (not used)", f"{r['min']:.1f}", f"{r['max']:.1f}",
      f"{r['mean']:.2f}", f"{r['std']:.2f}", f"{r['median']:.1f}"] for _, r in ds.iterrows()], "lrrrrr")

t_gini = table(f"Gini computation for the root split of a depth-1 tree ({tt(FEAT)} $\\leq {root['threshold']:.2f}$).", "tab:gini",
    ["Node", "Students", "$\\hat p$ (At-risk/Fail)", "Gini"],
    [["Root", nf(root["n_root"]), f4(root["p_fail_root"]), f4(root["gini_root"])],
     ["Left ($\\leq %.2f$ h)" % root["threshold"], nf(root["n_left"]), f4(root["p_fail_left"]), f4(root["gini_left"])],
     ["Right ($> %.2f$ h)" % root["threshold"], nf(root["n_right"]), f4(root["p_fail_right"]), f4(root["gini_right"])],
     ["Weighted children", nf(root["n_root"]), "---", f4(root["gini_weighted_children"])],
     ["Impurity decrease $\\Delta G$", "---", "---", f4(root["impurity_decrease"])]], "lrrr")

def diag_row(label, d):
    r = depth_df[depth_df.max_depth == d].iloc[0]
    gap = r.generalization_gap
    verdict = {1: "high on both: underfitting", d: ""}.get(d, "")
    return [label, str(d), f4(r.train_log_loss), f4(r.val_log_loss), f"{gap:+.4f}", f4(r.train_pr_auc), f4(r.val_pr_auc)]
dsel = [("A (very shallow)", 1), ("A$'$ (shallow)", 2), ("B (chosen)", best["max_depth"]), ("C (deep)", 10), ("C$'$ (very deep)", 15)]
t_diag = table("Training versus validation diagnosis for several tree depths (format of Example 8.3 in the lecture notes). "
    "$G = \\hat R_{\\mathrm{val}} - \\hat R_{\\mathrm{train}}$ uses log loss.", "tab:diag",
    ["Candidate", "Depth", "Train log loss", "Val. log loss", "Gap $G$", "Train PR-AUC", "Val. PR-AUC"],
    [diag_row(a, b) for a, b in dsel], "lrrrrrr")

def val_row(name, e):
    return [name, f"{e['threshold']:.2f}", f3(e["precision"]), f3(e["recall"]), f3(e["f1"]), f"{e['cost_per_student']:.3f}"]
t_thr = table("The two validation-selected thresholds (and the theoretical cost threshold) for the decision tree, evaluated on the validation set.", "tab:thr",
    ["Operating point", "$t$", "Precision", "Recall", "$F_1$", "Cost/student"],
    [val_row("$t_{F_1}$: maximises $F_1$", V["tree_F1_threshold"]), val_row("$t_{\\mathrm{cost}}$: minimises total cost", V["tree_cost_threshold"])], "lrrrrr")

t_test1 = table("Test-set classification metrics (positive class: At-risk/Fail). ``undef.'' means the denominator is zero (Sec. 12.4 of the lecture notes).", "tab:test1",
    ["Model / operating point", "$t$", "Acc.", "Bal. acc.", "Prec.", "Recall", "Spec.", "$F_1$"],
    [row_metrics("Majority class", mj), row_metrics("Logistic regression, $t_{F_1}$", lf), row_metrics("Logistic regression, $t_{\\mathrm{cost}}$", lcst),
     row_metrics("Decision tree, $t_{F_1}$", tf), row_metrics("Decision tree, $t_{\\mathrm{cost}}$", tc)], "lrrrrrrr")
t_test2 = table("Test-set threshold-free scores and mistake cost ($c_{FN}=%g$, $c_{FP}=%g$)." % (summary["cost_fn"], summary["cost_fp"]), "tab:test2",
    ["Model", "ROC-AUC", "PR-AUC", "Log loss", "Total cost", "Cost/student"],
    [["Majority class", f3(mj["roc_auc"]), f3(mj["pr_auc"]), f3(mj["log_loss"]), nf(mj["total_cost"]), f"{mj['cost_per_student']:.3f}"],
     ["Logistic regression", f3(lf["roc_auc"]), f3(lf["pr_auc"]), f3(lf["log_loss"]), "---", "---"],
     ["Decision tree", f3(tf["roc_auc"]), f3(tf["pr_auc"]), f3(tf["log_loss"]), "---", "---"],
     ["Logistic regression, $t_{\\mathrm{cost}}$", "", "", "", nf(lcst["total_cost"]), f"{lcst['cost_per_student']:.3f}"],
     ["Decision tree, $t_{F_1}$", "", "", "", nf(tf["total_cost"]), f"{tf['cost_per_student']:.3f}"],
     ["Decision tree, $t_{\\mathrm{cost}}$", "", "", "", nf(tc["total_cost"]), f"{tc['cost_per_student']:.3f}"]], "lrrrrr")

def cm_row(name, e):
    return [name, nf(e["TP"]), nf(e["FN"]), nf(e["FP"]), nf(e["TN"]), f3(e["fpr"]), f3(e["fnr"])]
t_cm = table("Confusion-matrix counts on the test set (Definition 12.1 layout) with false-positive and false-negative rates.", "tab:cm",
    ["Decision tree", "TP", "FN", "FP", "TN", "FPR", "FNR"],
    [cm_row("$t_{F_1} = %.2f$" % tf["threshold"], tf), cm_row("$t_{\\mathrm{cost}} = %.2f$" % tc["threshold"], tc)], "lrrrrrr")

n_te = tf["N"]
t_cost = table("Mean mistake cost per student on the test set for simple strategies and models.", "tab:cost",
    ["Strategy", "Cost / student", "Total cost"],
    [[k, f"{v:.3f}", nf(round(v * n_te))] for k, v in costs.items()], "lrr")

def sub_rows():
    rows = []
    for var in ["attendance_percentage", "class_participation"]:
        for g in ["low", "middle", "high"]:
            a = sub[(sub.variable == var) & (sub.group == g) & (sub.operating_point == "F1 threshold")].iloc[0]
            b = sub[(sub.variable == var) & (sub.group == g) & (sub.operating_point == "cost threshold")].iloc[0]
            rows.append([tt(var) if g == "low" else "", g, nf(a.n), pc(a.actual_fail_rate), f3(a.precision), f3(a.recall), f3(b.recall)])
    return rows
t_sub = table("Test-set error rates by tertile of two variables the tree barely uses (cut-offs from training data).", "tab:sub",
    ["Variable", "Group", "$n$", "Actual rate", "Prec. ($t_{F_1}$)", "Recall ($t_{F_1}$)", "Recall ($t_{\\mathrm{cost}}$)"],
    sub_rows(), "llrrrrr")

rule_lines = []
for _, r in leaves.iterrows():
    rule = (r.rule.replace("weekly_self_study_hours", "hours").replace("attendance_percentage", "attend.")
            .replace("class_participation", "partic.").replace(" AND ", " and ").replace("<=", r"$\leq$").replace(">", r"$>$").replace("<", r"$<$"))
    rule = rule.replace("$\\leq$ ", "$\\leq$ ").replace(" $>$ ", " $>$ ")
    rule_lines.append(f"{rule} & {nf(r.n_students)} & {r.p_fail:.3f} & {'FLAG' if r.flag_F1_threshold else 'ok'} & {'FLAG' if r.flag_cost_threshold else 'ok'} \\\\")
rules_table = ("\\begin{longtable}{p{7.3cm}rrcc}\n\\caption{All 32 leaves of the final decision tree (training data).}\\label{tab:rules}\\\\\n\\toprule\n"
               "Rule (conjunction of conditions) & Students & $\\hat p$ & $t_{F_1}$ & $t_{\\mathrm{cost}}$ \\\\\n\\midrule\n\\endfirsthead\n"
               "\\toprule\nRule & Students & $\\hat p$ & $t_{F_1}$ & $t_{\\mathrm{cost}}$ \\\\\n\\midrule\n\\endhead\n"
               + "\n".join(rule_lines) + "\n\\bottomrule\n\\end{longtable}\n")

lc_small, lc_big = lc.iloc[2], lc.iloc[-1]
sub_f1 = sub[sub.operating_point == "F1 threshold"]
rec_min, rec_max = sub_f1.recall.min(), sub_f1.recall.max()
cv = metrics["cv_stability"]
bayes = metrics["bayes_check"]
spl = metrics["splits_by_feature"]
cal_maxdiff = (cal.mean_predicted - cal.observed_rate).abs().max()
sel_gap = float(tune["best_mean_cv_score"] - tune["chosen_mean_cv_score"])
best_dep = depth_df.loc[depth_df.val_log_loss.idxmin()]
flag_f1_cut = leaves[leaves.flag_F1_threshold].rule.str.extract(r"<= ([\d.]+)")[0].astype(float)
feat_hours_flag = f"{tf['threshold']:.2f}"
share_flag_f1, share_flag_cost = (tf["TP"] + tf["FP"]) / tf["N"], (tc["TP"] + tc["FP"]) / tc["N"]

V_ = dict(
 TITLE="Predicting Student At-Risk Status from Behavioral Study Data: A Decision Tree Classifier for Early Identification of Struggling Students",
 NROWS=nf(summary["n_rows"]), FAILRATE=pc(summary["fail_rate"]), NFAIL=nf(n_fail),
 NTRAIN=nf(summary["n_train"]), NVAL=nf(summary["n_val"]), NTEST=nf(summary["n_test"]),
 DEPTH=best["max_depth"], LEAVES=bp["n_leaves"], CRIT=best["criterion"], MSL=best["min_samples_leaf"],
 TF1=f"{th['t_f1']:.2f}", TCOST=f"{th['t_cost']:.2f}", TTHEORY=f"{th['t_theory']:.3f}",
 CFN=f"{summary['cost_fn']:g}", CFP=f"{summary['cost_fp']:g}", SEED=summary["random_state"],
 ACC=f3(tf["accuracy"]), BACC=f3(tf["balanced_accuracy"]), PREC=f3(tf["precision"]), REC=f3(tf["recall"]), SPEC=f3(tf["specificity"]),
 F1=f3(tf["f1"]), ROC=f3(tf["roc_auc"]), PRAUC=f3(tf["pr_auc"]),
 ACC_C=f3(tc["accuracy"]), BACC_C=f3(tc["balanced_accuracy"]), PREC_C=f3(tc["precision"]), REC_C=f3(tc["recall"]), F1_C=f3(tc["f1"]),
 LR_F1=f3(lf["f1"]), LR_PR=f3(lf["pr_auc"]), LR_ROC=f3(lf["roc_auc"]), LR_REC=f3(lf["recall"]), LR_PREC=f3(lf["precision"]),
 MJ_ACC=f3(mj["accuracy"]),
 COST_NONE=f"{costs['Flag nobody']:.3f}", COST_ALL=f"{costs['Flag everybody']:.3f}", COST_TF1=f"{costs['Decision tree (F1 threshold)']:.3f}",
 COST_TC=f"{costs['Decision tree (cost threshold)']:.3f}", COST_LRC=f"{costs['Logistic regression (cost threshold)']:.3f}",
 COST_SAVE=pc(1 - costs["Decision tree (cost threshold)"] / costs["Flag nobody"], 0),
 CVAP=f3(cv["average_precision"][0]), CVAPSD=f3(cv["average_precision"][1]), CVROC=f3(cv["roc_auc"][0]), CVROCSD=f3(cv["roc_auc"][1]),
 BAYES=f4(bayes["bayes_precision"]), EMP=f4(bayes["empirical_precision"]),
 SPL_H=spl["weekly_self_study_hours"], SPL_A=spl["attendance_percentage"], SPL_P=spl["class_participation"],
 IMP_H=f"{100 * imp.loc[FEAT, 'gini_importance']:.2f}\\%", IMP_A=f"{100 * imp.loc['attendance_percentage', 'gini_importance']:.3f}\\%",
 IMP_P=f"{100 * imp.loc['class_participation', 'gini_importance']:.3f}\\%",
 PERM_H=f"{imp.loc[FEAT, 'permutation_importance']:.3f}", PERM_A=f"{imp.loc['attendance_percentage', 'permutation_importance']:+.4f}",
 TUNE_BEST=f4(tune["best_mean_cv_score"]), TUNE_SE=f4(tune["standard_error"]), TUNE_CHOSEN=f4(tune["chosen_mean_cv_score"]),
 NCONF=tune["n_configurations"], NFITS=tune["n_fits"], NTUNE=nf(tune["tuning_rows"]),
 LC_SMALL_N=nf(lc_small.n_train), LC_SMALL_G=f"{lc_small.val_log_loss - lc_small.train_log_loss:.4f}",
 LC_BIG_G=f"{lc_big.val_log_loss - lc_big.train_log_loss:+.4f}",
 BESTDEP=int(best_dep.max_depth), BESTDEP_LL=f4(best_dep.val_log_loss),
 CALMAX=f"{cal_maxdiff:.3f}", RECMIN=f3(rec_min), RECMAX=f3(rec_max),
 SHARE100=pc(summary["share_total_score_100"]), MED_P=f"{summary['median_study_hours']['pass']:.1f}", MED_F=f"{summary['median_study_hours']['fail']:.1f}",
 CORR_H=f"{summary['corr_with_total_score'][FEAT]:.3f}", CORR_A=f"{summary['corr_with_total_score']['attendance_percentage']:.4f}",
 CORR_P=f"{summary['corr_with_total_score']['class_participation']:.4f}",
 CORRF_A=f"{summary['corr_with_fail']['attendance_percentage']:.4f}", CORRF_P=f"{summary['corr_with_fail']['class_participation']:.4f}",
 FLAGSHARE_F1=pc(share_flag_f1), FLAGSHARE_C=pc(share_flag_cost),
 ROOT_T=f"{root['threshold']:.2f}", ROOT_PR=f4(root["p_fail_root"]), ROOT_G=f4(root["gini_root"]), ROOT_GW=f4(root["gini_weighted_children"]), ROOT_DG=f4(root["impurity_decrease"]),
 ROOT_PL=f4(root["p_fail_left"]), ROOT_PRR=f4(root["p_fail_right"]),
 RUNTIME=f"{summary['runtime_seconds']:.0f}",
)
TOK = dict(T_DESC=t_desc, T_GINI=t_gini, T_DIAG=t_diag, T_THR=t_thr, T_TEST1=t_test1, T_TEST2=t_test2, T_CM=t_cm, T_COST=t_cost, T_SUB=t_sub, RULES_TABLE=rules_table)
TOK["F_CLASS"] = figure("01_class_distribution.png", "Letter-grade distribution (left) and the binary target with the positive class $y=1$ (right).", "fig:class", "0.85")
TOK["F_FEAT"] = figure("02_feature_distributions.png", "Feature distributions by outcome (density histograms).", "fig:feat", "0.98")
TOK["F_CORR"] = figure("04_correlation_heatmap.png", "Pearson correlation matrix. \\texttt{total\\_score} is shown for reference only; it is not a model input.", "fig:corr", "0.55")
TOK["F_SCAT"] = figure("05_study_hours_vs_score.png", "Study hours versus total score for a random sample of 6,000 students. The dashed line is the At-risk/Fail cutoff (score $=55$).", "fig:scat", "0.7")
TOK["F_IMP"] = figure("06_impurity_functions.png", "Gini impurity and entropy as functions of the class proportion $p$ in a binary node.", "fig:imp", "0.6")
TOK["F_SPLIT"] = figure("07_data_split.png", "Stratified three-way split (same idea as Figure~2 of the lecture notes).", "fig:split", "0.95")
TOK["F_GRID"] = figure("08_grid_search_heatmap.png", "GridSearchCV results: mean 5-fold CV PR-AUC by \\texttt{max\\_depth} and \\texttt{min\\_samples\\_leaf}.", "fig:grid", "0.8")
TOK["F_DEPTH"] = figure("09_depth_diagnosis.png", "Training versus validation log loss and PR-AUC as model capacity (tree depth) grows.", "fig:depth", "0.98")
TOK["F_LC"] = figure("10_learning_curve.png", "Learning curves of the chosen configuration: performance against training-set size.", "fig:lc", "0.98")
TOK["F_THR"] = figure("13_threshold_and_cost.png", "Validation-set study of the threshold $t$: precision/recall/$F_1$ (left) and mistake cost per student (right).", "fig:thr", "0.98")
TOK["F_CAL"] = figure("14_calibration.png", "Reliability diagram of the tree on the validation set.", "fig:cal", "0.5")
TOK["F_CM"] = figure("11_confusion_matrices.png", "Test-set confusion matrices at the two validation-selected thresholds.", "fig:cm", "0.95")
TOK["F_ROC"] = figure("12_roc_pr_curves.png", "ROC and precision--recall curves on the test set.", "fig:roc", "0.95")
TOK["F_COMP"] = figure("18_model_comparison.png", "Test-set comparison of the baselines and the decision tree.", "fig:comp", "0.9")
TOK["F_COST"] = figure("19_cost_comparison.png", "Mean mistake cost per student on the test set.", "fig:cost", "0.85")
TOK["F_TREE"] = figure("16_decision_tree_depth3.png", "Illustrative depth-3 tree trained on the same data (the final model has depth %d)." % best["max_depth"], "fig:tree", "1.0")
TOK["F_IMPORT"] = figure("15_feature_importance.png", "Feature importance: impurity-based (left) and permutation importance on the test set (right).", "fig:fimp", "0.95")
TOK["F_SURF"] = figure("17_decision_surface.png", "Decision surface: predicted probability of At-risk/Fail by study hours and attendance.", "fig:surf", "0.65")

TEMPLATE = r"""\documentclass[12pt,a4paper]{article}

\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{setspace}
\usepackage{times}
\usepackage{titlesec}
\usepackage{amsmath, amssymb, amsthm}
\usepackage{booktabs}
\usepackage{array}
\usepackage{hyperref}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage{indentfirst}
\usepackage{longtable}
\usepackage{listings}
\usepackage{float}

\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=black,
    urlcolor=blue
}

\onehalfspacing
\setlength{\parindent}{0.5in}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\graphicspath{{../figures/}{assets/}}

\titleformat{\section}
    {\normalfont\bfseries\large}{\thesection.}{0.5em}{}
\titleformat{\subsection}
    {\normalfont\bfseries}{\thesubsection.}{0.5em}{}

\lstset{basicstyle=\ttfamily\footnotesize, breaklines=true, frame=single, backgroundcolor=\color[gray]{0.96},
        columns=fullflexible, keepspaces=true, xleftmargin=0.2in, xrightmargin=0.1in}
\usepackage{xcolor}

\theoremstyle{definition}
\newtheorem{definition}{Definition}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{Decision Tree for At-Risk Student Prediction}
\fancyhead[R]{\thepage}
\setlength{\headheight}{14.5pt}
\renewcommand{\headrulewidth}{0.4pt}

\newcommand{\UniversityName}{Bulacan State University}
\newcommand{\CollegeName}{College of Science}
\newcommand{\CourseCode}{MCS 306}
\newcommand{\CourseDescription}{Introduction to Artificial Intelligence}
\newcommand{\AcademicYear}{Academic Year 2026--2027}
\newcommand{\StudyTitle}{@@TITLE@@}
\newcommand{\AuthorNames}{Calug, Adrian Samuel \\ Hilario, Angelo \\ Jaculina, Adrienne Christopher \\ Mendoza, Clarence Dion Reigh \\ Yumul, Aries}
\newcommand{\CourseYearSection}{BS Mathematics with Specialization in Computer Science \\ 3rd Year -- Section B}
\newcommand{\InstructorName}{Aarhus M. Dela Cruz, LPT, MSc}
\newcommand{\SubmissionDate}{October 27, 2026}

\newcommand{\BulSULogo}{assets/bulsu.png}
\newcommand{\COSLogo}{assets/college_of_science.png}

\begin{document}

\begin{titlepage}
    \centering
    \singlespacing

    \vspace*{0.3in}

    \begin{minipage}{0.22\textwidth}
        \centering
        \includegraphics[width=0.95\linewidth]{\BulSULogo}
    \end{minipage}
    \hfill
    \begin{minipage}{0.50\textwidth}
        \centering
        {\Large\bfseries \UniversityName\par}
        \vspace{0.15cm}
        {\large\bfseries \CollegeName\par}
        \vspace{0.15cm}
        {\normalsize \CourseCode: \CourseDescription\par}
        \vspace{0.15cm}
        {\normalsize \AcademicYear\par}
    \end{minipage}
    \hfill
    \begin{minipage}{0.22\textwidth}
        \centering
        \includegraphics[width=0.95\linewidth]{\COSLogo}
    \end{minipage}

    \vspace{0.55in}

    {\Large\bfseries MINI RESEARCH AND EXPOSITION PAPER\par}

    \vspace{0.8in}

    {\Large\bfseries \StudyTitle\par}

    \vspace{0.4in}

    {\large\bfseries Authors\par}
    \vspace{0.2cm}
    {\large \AuthorNames\par}

    \vspace{0.35in}

    {\large \CourseYearSection\par}

    \vfill

    {\normalsize Submitted to:\par}
    \vspace{0.15cm}
    {\normalsize\bfseries \InstructorName\par}

    \vspace{0.25in}

    {\normalsize \SubmissionDate\par}
\end{titlepage}

\pagenumbering{roman}
\setcounter{page}{1}

\begin{abstract}
\noindent Teachers handling large classes rarely notice struggling students before final grades are released. This paper builds an interpretable early-warning model that predicts whether a student will finish At-risk/Fail (letter grade D or F) from three behavioral variables: weekly self-study hours, class attendance, and class participation. Following the model-development workflow of the course, a CART Decision Tree Classifier is trained on the Kaggle Student Performance Dataset (@@NROWS@@ students, @@FAILRATE@@ At-risk/Fail) using a stratified training/validation/test split. Hyperparameters are chosen by five-fold cross-validated grid search, decision thresholds are chosen on the validation set only, and the test set is used once. On @@NTEST@@ unseen students, the final tree (depth @@DEPTH@@, @@LEAVES@@ leaves) reaches ROC-AUC @@ROC@@. At the $F_1$-optimal threshold it attains precision @@PREC@@ and recall @@REC@@; at a cost-sensitive threshold (missed student costs @@CFN@@ units, false alarm @@CFP@@) recall rises to @@REC_C@@ and the mistake cost falls by @@COST_SAVE@@ compared with flagging no one. The tree performs on par with a logistic-regression baseline while remaining readable as if--then rules. Weekly self-study hours is the only informative predictor; attendance and participation show essentially no association with the outcome, which together with other features of the data suggests that the dataset is simulated.
\end{abstract}

\noindent\textbf{Keywords:} decision tree, early-warning system, student performance, imbalanced classification, threshold selection

\newpage
\tableofcontents
\newpage

\pagenumbering{arabic}
\setcounter{page}{1}

\section{Introduction}

In large classes, an instructor may be responsible for hundreds of students. By the time midterm or final results are released, the chance to help students who are struggling has often passed. An \emph{early-warning system} that flags students at risk, using information that teachers can collect during the semester, could allow earlier consultation, tutoring, or referral. This supports the United Nations Sustainable Development Goal~4 (Quality Education), which stresses inclusive and equitable learning.

The problem fits the idea of a learning agent discussed in Chapter~9 of the course lecture notes. Following Mitchell's operational description of learning \cite{mitchell1997}, the \emph{task} $T$ is to predict whether a student will end the course with a grade of D or F; the \emph{experience} $E$ is a set of past student records with their eventual outcomes; and the \emph{performance measure} $P$ is a set of metrics on unseen students, namely precision, recall, $F_1$, and a cost-weighted measure. As the lecture notes stress, a prediction is not yet a decision: the agent must still choose whether to flag a student, and that choice depends on the consequences of false alarms and missed cases.

A decision tree is a natural model for this agent. It produces human-readable if--then rules, needs no feature scaling, and is a standard introductory supervised model \cite{russell2022}. Transparency matters in education because teachers and students should be able to understand \emph{why} a student was flagged.

\subsection{Problem Statement and Significance}

The study asks: \emph{How well can a decision tree, using only behavioral variables available before final grades, identify students who will end with a D or F, and how should its alert threshold be chosen?} The significance is practical (a simple rule that a teacher can apply by hand) and methodological (an end-to-end, leak-free, reproducible machine-learning workflow that follows the lecture notes).

\subsection{Objectives of the Paper}

The objectives of this paper are as follows:

\begin{enumerate}[label=\arabic*.]
    \item To formulate student At-risk/Fail prediction as a supervised binary classification problem with a clearly defined positive class;
    \item To explain the decision-tree model, its impurity measures, and the evaluation concepts (confusion matrix, thresholds, calibration, mistake cost) used in the course;
    \item To train, tune, and evaluate a Decision Tree Classifier with a training/validation/test protocol and compare it with a majority-class rule and a logistic-regression baseline; and
    \item To interpret the learned rules, discuss limitations and ethical issues, and suggest future directions.
\end{enumerate}

\subsection{Scope and Limitations}

The study uses a single public dataset of @@NROWS@@ records and three behavioral predictors. The dataset contains no demographic variables (such as age, sex, or family background), no prior grades, and no timestamps. Therefore the model is behavioral only, and a time-aware evaluation (Sec.~7.2 of the notes) is not possible. Findings describe associations in this dataset and are not causal claims.

\section{Review of Related Literature and Background Concepts}

\subsection{Key Definitions}

The definitions below follow Chapters~7--9 of the lecture notes.

\begin{definition}[Supervised learning and the labeled training set]
Supervised learning uses examples containing inputs $x$ and desired outputs $y$ to learn a mapping that predicts $y$ for unseen inputs. A labeled training set of $n$ observations is
$\mathcal{D}_{\mathrm{train}} = \{(x_i, y_i) : i = 1,\dots,n\}$. Here $x_i = (\text{study}_i, \text{attendance}_i, \text{participation}_i)$ and $y_i = 1$ if student $i$ ended with a D or F (the \emph{positive class}) and $y_i = 0$ otherwise.
\end{definition}

A model $h_\theta$ maps features to a prediction $\hat y = h_\theta(x)$. \emph{Parameters} $\theta$ are learned from data (for a tree, the split rules), whereas \emph{hyperparameters} such as the maximum depth are chosen before fitting and selected with validation data. With a loss function $\ell(y,\hat y)$, the \emph{empirical risk} is the average training loss,
\begin{equation}
    \hat R_{\mathcal{D}_{\mathrm{train}}}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell\bigl(y_i, h_\theta(x_i)\bigr).
\end{equation}
A classifier usually estimates $\hat p(x) \approx P(Y=1 \mid X=x)$ and converts it to a class by a \emph{threshold} $t$:
\begin{equation}
    \hat y_t = \begin{cases} 1, & \hat p(x) \ge t,\\ 0, & \hat p(x) < t. \end{cases}
\end{equation}
\emph{Generalization} is useful performance on new, unseen examples; \emph{underfitting} means inadequate fit even on training data, and \emph{overfitting} means fitting accidental details so that validation performance is poor. The \emph{generalization gap} is $G = \hat R_{\mathrm{val}}(h) - \hat R_{\mathrm{train}}(h)$.

\paragraph{Decision trees.} A decision tree partitions the feature space with questions of the form ``feature $j \le s$''. Each leaf stores the fraction of training students that failed, which serves as $\hat p(x)$. CART grows the tree greedily \cite{breiman1984}, choosing at each node the split that most reduces impurity. For a binary node with a fraction $p$ of positive cases, the Gini impurity and the entropy are
\begin{align}
    G(p) &= 1 - p^2 - (1-p)^2, \\
    H(p) &= -p\log_2 p - (1-p)\log_2(1-p).
\end{align}
Both are $0$ for a pure node and largest at $p=0.5$ (Figure~\ref{fig:imp}). If a split sends $n_L$ of $n$ students to the left child and $n_R$ to the right, the impurity decrease is
\begin{equation}
    \Delta G = G(p) - \Bigl[\tfrac{n_L}{n} G(p_L) + \tfrac{n_R}{n} G(p_R)\Bigr]. \label{eq:dg}
\end{equation}
@@F_IMP@@

\paragraph{Evaluation metrics.} With the confusion-matrix counts TP, FN, FP, TN (actual positive row first, as in the notes),
\begin{align}
    \text{Accuracy} &= \tfrac{TP+TN}{TP+TN+FP+FN}, &
    \text{Precision} &= \tfrac{TP}{TP+FP}, &
    \text{Recall} &= \tfrac{TP}{TP+FN},\\
    \text{Specificity} &= \tfrac{TN}{TN+FP}, &
    \text{FPR} &= \tfrac{FP}{FP+TN}, &
    \text{FNR} &= \tfrac{FN}{FN+TP},\\
    F_1 &= \tfrac{2\,TP}{2\,TP+FP+FN}, &
    \text{Bal. accuracy} &= \tfrac{1}{2}(\text{Recall}+\text{Specificity})
\end{align}
Because the positive class is rare, accuracy can be misleading; precision, recall, $F_1$, balanced accuracy, and the area under the precision--recall curve (average precision) are more informative \cite{saito2015}. The log loss
$-\frac{1}{N}\sum_i [y_i\ln p_i + (1-y_i)\ln(1-p_i)]$ measures the quality of predicted probabilities, and \emph{calibration} asks whether, among students given probability near $q$, about a fraction $q$ really fail.

\paragraph{Cost-sensitive decisions.} If flagging a student who would pass costs $c_{FP}$ and missing a student who fails costs $c_{FN}$, and $\hat p$ is calibrated, the expected costs are $c_{FP}(1-\hat p)$ for flagging and $c_{FN}\hat p$ for not flagging. Flagging has the lower expected cost exactly when
\begin{equation}
    \hat p \;\ge\; \frac{c_{FP}}{c_{FP}+c_{FN}}, \label{eq:costthr}
\end{equation}
which is the maximum-expected-utility principle of Chapter~7 applied to two actions (Theorem~15.2 of Chapter~9). The total mistake cost on a test set is $C = c_{FN}\,FN + c_{FP}\,FP$.

\subsection{Related Studies}

Educational data mining applies machine-learning methods to educational data in order to understand and improve learning; reviews of the field describe prediction of student performance as one of its core tasks \cite{romero2010,baker2009}. A frequently cited study on secondary-school data compared several data-mining methods, including decision trees, and reported that information about earlier grades was highly predictive of the final grade \cite{cortez2008}. Such results suggest that behavioral variables alone may be weaker predictors than prior achievement, which is relevant to how the results of this paper should be read. Decision trees are attractive in this setting because their rules can be inspected \cite{breiman1984,quinlan1986}. Methodologically, work on imbalanced classification recommends precision--recall analysis over ROC analysis alone when positives are rare \cite{saito2015}, and work on data leakage warns against using information that would not be available at prediction time \cite{kaufman2012}. The present paper combines these ideas in one reproducible, course-aligned workflow.

\section{Methodology}

\subsection{Research Design and Workflow}

The study is a supervised, quantitative, predictive modeling study. The code implements the eight-step workflow of Section~11.5 of the lecture notes: (1) define the target, the positive class, and the cost criterion; (2) reserve the final test set with a stratified split; (3) inspect development data and build simple baselines; (4) define pipelines and candidate estimators; (5) use cross-validation and validation data to select hyperparameters and thresholds; (6) inspect errors using development data only; (7) freeze the final selection rule and model; and (8) evaluate once on the untouched test set.

\subsection{Dataset and Variables}

The Student Performance Dataset \cite{kaggle} contains @@NROWS@@ student records with no missing values and no duplicate identifiers (Table~\ref{tab:desc}). The variables are \texttt{weekly\_self\_study\_hours} (0--40), \texttt{attendance\_percentage} (50--100), \texttt{class\_participation} (0--10), \texttt{total\_score} (0--100), and the letter \texttt{grade} (A--F), which is derived from the total score (approximately A $\ge 85$, B $\ge 70$, C $\ge 55$, D $\ge 40$, F $< 40$).

@@T_DESC@@

\subsection{Target Variable and Leakage Control}

The positive class is \emph{At-risk/Fail}: $y=1$ if the grade is D or F (total score below 55) and $y=0$ for grades A--C. This gives @@NFAIL@@ positive students (@@FAILRATE@@; Figure~\ref{fig:class}), so the classes are imbalanced. The cutoff is a modeling choice and can be changed in the code.

@@F_CLASS@@

The predictors are the three behavioral variables. \texttt{total\_score} is deliberately \emph{excluded}: the grade is computed from it, so using it would be target leakage (Definition~7.2 and Activity~18.2 of the notes \cite{kaufman2012}) and would not be available before the final result. \texttt{student\_id} is excluded because it is only an identifier. Because the decision tree is invariant to monotone rescaling of features, no scaler is used for the tree; the logistic-regression baseline learns its scaler inside the pipeline.

\subsection{Data Splitting}

Following Definition~7.1 of the notes, the data are divided by stratified sampling (random seed @@SEED@@) into training (@@NTRAIN@@ students), validation (@@NVAL@@), and test (@@NTEST@@) sets (Figure~\ref{fig:split}). The \textbf{test set is reserved first} and used only once at the end. The training set fits the model and is used for cross-validated tuning; the validation set is used only to select thresholds and to diagnose fitting.

@@F_SPLIT@@

\subsection{Model or Method Description}

The model is a CART classification tree: a function $h_\theta$ defined by a binary tree of axis-aligned splits (``feature $j \le s$''), where the learned parameters $\theta$ are the chosen features, split thresholds, and leaf class proportions. Training is empirical-risk minimization carried out greedily: at each node, all features and candidate thresholds are scanned and the split with the largest impurity decrease $\Delta G$ (Equation~\ref{eq:dg}) is taken. The recursion stops at the maximum depth or when a split would create a leaf with fewer than the minimum number of samples.

\emph{Assumptions.} (i) Students are independent and identically distributed (Sec.~6.3 of the notes), which is not guaranteed for real classes sharing instructors or terms; (ii) the three predictors are measurable before final grades; (iii) the relationship between habits and outcomes is stable between training and deployment.

\subsection{Implementation and Code}

The code (Python 3, scikit-learn, pandas, matplotlib; full code in the Git repository) imitates the style of the coding examples in Section~19 of the notes: a \texttt{Pipeline}, a stratified five-fold \texttt{GridSearchCV}, and an untouched test set.

\begin{lstlisting}[language=Python]
pipeline = Pipeline([("model",
            DecisionTreeClassifier(random_state=11))])
cv = StratifiedKFold(n_splits=5, shuffle=True,
                     random_state=11)
search = GridSearchCV(pipeline, param_grid,
                      scoring="average_precision",
                      cv=cv, refit=False)
search.fit(X_tr_sub, y_tr_sub)

tree = Pipeline([("model", DecisionTreeClassifier(
        criterion="@@CRIT@@", max_depth=@@DEPTH@@,
        min_samples_leaf=@@MSL@@, random_state=11))])
tree.fit(X_train, y_train)
p_val = tree.predict_proba(X_val)[:, 1]
t_f1, t_cost = select_thresholds(
        threshold_scan(y_val, p_val))
p_test = tree.predict_proba(X_test)[:, 1]
y_pred = (p_test >= t_f1).astype(int)
\end{lstlisting}

\paragraph{Hyperparameter search.} The grid contains @@NCONF@@ configurations (criterion $\in$ \{gini, entropy\}; \texttt{max\_depth} $\in$ \{2,3,4,5,6,8,10,12\}; \texttt{min\_samples\_leaf} $\in$ \{1,50,500,2000\}), evaluated by five-fold stratified cross-validation (@@NFITS@@ fits) on a stratified subsample of @@NTUNE@@ training students for speed. The selection metric is the cross-validated average precision (PR-AUC), chosen because it reflects the precision--recall trade-off for a rare positive class (Sec.~12.9). Following Example~11.2 (small differences between fold scores do not establish superiority) and Section~10.2 (prefer simpler hypotheses), the simplest configuration within one standard error of the best mean score was selected.

\paragraph{Step size, iterations, and tolerance.} A decision tree is not trained by iterative numerical optimization, so there is no step size or convergence tolerance. Training is a deterministic greedy construction whose size is governed by \texttt{max\_depth} (@@DEPTH@@), \texttt{min\_samples\_leaf} (@@MSL@@), and the minimum impurity decrease (left at the default, $0$). Estimation error is quantified by cross-validation (Sec.~\ref{sec:results}).

\paragraph{Thresholds.} Two operating points are selected using validation data only (never the test set): $t_{F_1}$, which maximizes $F_1$, and $t_{\mathrm{cost}}$, which minimizes the total cost $C = @@CFN@@\,FN + @@CFP@@\,FP$. The costs are the illustrative values used in the lecture-note example (a missed case costs @@CFN@@ units, an unnecessary flag @@CFP@@ units); a school should replace them by its own estimates. Equation~\ref{eq:costthr} gives the theoretical threshold @@TTHEORY@@ for these costs.

\paragraph{Baselines and checks.} A majority-class rule and a standardized logistic regression are fitted and thresholded by the same protocol (Sec.~7.5). All metric functions are implemented directly from the definitions in the notes and verified against the worked examples printed there (for instance Example~12.8 and Example~15.1) by an automated test file in the repository.

\section{Main Discussion, Results, and Analysis}\label{sec:results}

\subsection{Exploratory Analysis}

Figures~\ref{fig:feat}--\ref{fig:scat} show that the two groups differ sharply in study hours (median @@MED_F@@ hours per week for At-risk/Fail students versus @@MED_P@@ for the others) but are almost indistinguishable in attendance and participation. The correlation of study hours with the total score is @@CORR_H@@, while attendance and participation correlate with the total score at @@CORR_A@@ and @@CORR_P@@, and with the outcome at @@CORRF_A@@ and @@CORRF_P@@.

@@F_FEAT@@
@@F_CORR@@
@@F_SCAT@@

\subsection{Illustrative Example: One Split by Hand}

Table~\ref{tab:gini} reproduces the root split of a depth-1 tree on the training data. The best split is $\text{hours} \le @@ROOT_T@@$. Using $G(p)=1-p^2-(1-p)^2$ with $p=@@ROOT_PR@@$ gives $G=@@ROOT_G@@$; the size-weighted child impurity is @@ROOT_GW@@, so $\Delta G = @@ROOT_G@@ - @@ROOT_GW@@ = @@ROOT_DG@@$ (Equation~\ref{eq:dg}). The left child is nearly a coin flip ($\hat p=@@ROOT_PL@@$), whereas the right child is mostly passing ($\hat p=@@ROOT_PRR@@$). The same value is reported by scikit-learn, which confirms the hand-written formula.

@@T_GINI@@

\subsection{Model Selection and Fit Diagnosis}

The grid search (Figure~\ref{fig:grid}) found its best mean cross-validated PR-AUC of @@TUNE_BEST@@ (standard error @@TUNE_SE@@). Several configurations are within one standard error, so the simplest was chosen: \texttt{criterion}~=~@@CRIT@@, \texttt{max\_depth}~=~@@DEPTH@@, \texttt{min\_samples\_leaf}~=~@@MSL@@ (mean CV PR-AUC @@TUNE_CHOSEN@@, a loss of only @@SEL_GAP@@). The final tree has @@LEAVES@@ leaves.

@@F_GRID@@

Training and validation results across depths (Figure~\ref{fig:depth}, Table~\ref{tab:diag}) follow the pattern described in Example~8.3 of the notes. A depth-1 tree has high log loss on \emph{both} sets, which is consistent with underfitting: a single split cannot express the gradual decline of failure probability with study hours. Between depths 4 and 5 the validation log loss is smallest (@@BESTDEP_LL@@ at depth @@BESTDEP@@) and the gap $G$ is essentially zero. Beyond that point the training risk keeps falling while the validation risk rises, so $G$ grows, which is the signature of overfitting.

@@T_DIAG@@
@@F_DEPTH@@

The learning curve of the chosen configuration (Figure~\ref{fig:lc}) shows that the gap between training and validation log loss is large for small samples (@@LC_SMALL_G@@ at @@LC_SMALL_N@@ students) and closes as data grow (@@LC_BIG_G@@ with all training data). Both curves flatten, so more data of the same kind would not help much; the remaining error reflects limited features and noise (Sec.~8.4).

@@F_LC@@

\subsection{Choosing the Operating Threshold on Validation Data}

The tree outputs a probability $\hat p$ (the failure rate of the leaf), and the alert rule is $\hat p \ge t$. Figure~\ref{fig:thr} (left) shows the familiar trade-off: lowering $t$ raises recall and lowers precision. The $F_1$-optimal threshold on the validation set is $t_{F_1}=@@TF1@@$. Figure~\ref{fig:thr} (right) shows the mistake cost per student; its minimum is at $t_{\mathrm{cost}}=@@TCOST@@$, practically equal to the theoretical value @@TTHEORY@@ from Equation~\ref{eq:costthr}. This agreement is expected only if the probabilities are calibrated; the reliability diagram in Figure~\ref{fig:cal} confirms this (largest gap between mean predicted probability and observed rate across bins: @@CALMAX@@).

@@T_THR@@
@@F_THR@@
@@F_CAL@@

\subsection{Test-Set Results}

With every choice frozen, the models were evaluated once on the @@NTEST@@ test students (Tables~\ref{tab:test1}--\ref{tab:cm}, Figures~\ref{fig:cm}--\ref{fig:comp}).

@@T_TEST1@@

At the $F_1$-optimal threshold the tree attains accuracy @@ACC@@, but accuracy is dominated by the large Pass class: a majority-class rule already reaches @@MJ_ACC@@ while detecting no one (recall $0$, precision undefined because nothing is flagged). The more informative figures are precision @@PREC@@, recall @@REC@@, specificity @@SPEC@@, $F_1$ @@F1@@, and balanced accuracy @@BACC@@. At the cost-optimal threshold the tree flags more students (@@FLAGSHARE_C@@ of all students versus @@FLAGSHARE_F1@@) and raises recall to @@REC_C@@ at the price of lower precision (@@PREC_C@@).

@@T_CM@@
@@F_CM@@

As a consistency check with Chapter~8, Bayes' rule applied to the prevalence, the recall (as $P(E\mid H)$), and the false-positive rate (as $P(E\mid \neg H)$) gives a precision of @@BAYES@@, identical to the empirical precision @@EMP@@: precision depends on prevalence, which is why it is modest although the false-positive rate is small.

@@T_TEST2@@

The threshold-free scores (Table~\ref{tab:test2}) are nearly identical for the tree and the logistic regression (ROC-AUC @@ROC@@ versus @@LR_ROC@@; PR-AUC @@PRAUC@@ versus @@LR_PR@@), and the curves in Figure~\ref{fig:roc} overlap. Cross-validation on the full training set gave PR-AUC @@CVAP@@ $\pm$ @@CVAPSD@@ and ROC-AUC @@CVROC@@ $\pm$ @@CVROCSD@@ (mean $\pm$ SD over folds), so the model is stable.

@@F_ROC@@
@@F_COMP@@

\paragraph{Decision quality.} Prediction quality is not decision quality (Sec.~15.1). With $c_{FN}=@@CFN@@$ and $c_{FP}=@@CFP@@$, flagging nobody costs @@COST_NONE@@ units per student and flagging everybody costs @@COST_ALL@@. The tree at $t_{F_1}$ costs @@COST_TF1@@ and at $t_{\mathrm{cost}}$ @@COST_TC@@ (logistic regression at its cost threshold: @@COST_LRC@@; Table~\ref{tab:cost}, Figure~\ref{fig:cost}). The cost-optimal tree therefore reduces the mistake cost by @@COST_SAVE@@ compared with doing nothing, even though its $F_1$ and accuracy are lower than at $t_{F_1}$. This illustrates that the better metric depends on the consequences of the errors, and that these costs are assumptions, not measurements.

@@T_COST@@
@@F_COST@@

\subsection{Interpretation of the Tree}

The final tree uses the study-hours feature in @@SPL_H@@ of its splits, attendance in @@SPL_A@@, and participation in @@SPL_P@@. Its impurity-based importances are @@IMP_H@@ for study hours, @@IMP_A@@ for attendance, and @@IMP_P@@ for participation, and permutation importance on the test set (decrease in PR-AUC) is @@PERM_H@@ for study hours but @@PERM_A@@ for attendance, i.e., shuffling attendance does not hurt (Figure~\ref{fig:fimp}). The few attendance and participation splits are therefore fitting random fluctuation inside the data, not real structure---a small illustration of overfitting. Figure~\ref{fig:tree} shows an illustrative depth-3 tree, and Appendix~\ref{app:rules} lists all rules. In practical terms, the rule at $t_{F_1}$ flags students who study about 4 hours per week or less (probability of failing above @@TF1@@), and the rule at $t_{\mathrm{cost}}$ extends this to students studying up to roughly 6 hours.

@@F_TREE@@
@@F_IMPORT@@
@@F_SURF@@

\subsection{Error Rates Across Groups}

Aggregate performance can hide unequal errors (Sec.~15.2). Table~\ref{tab:sub} reports test-set metrics by tertile of attendance and of participation. Recall at $t_{F_1}$ ranges only from @@RECMIN@@ to @@RECMAX@@ across the six groups, and the flag rate and precision are also similar, so the tree treats low- and high-attendance students alike. This is expected because the tree hardly uses these variables; it does not rule out unfairness with respect to variables that are not in the data.

@@T_SUB@@

\subsection{Analysis and Interpretation}

\paragraph{Accuracy and stability.} The tree separates at-risk from other students well (ROC-AUC @@ROC@@), but exact identification is limited (precision about one half at $t_{F_1}$). Even among students who study less than one hour per week, a sizable fraction still pass (Appendix~\ref{app:rules}); the final score depends on factors not recorded here. The tree is \emph{not} better than logistic regression (F$_1$ @@F1@@ versus @@LR_F1@@), which is plausible because the true relationship is a smooth monotone function of study hours that a logistic curve represents more compactly than a step function. The advantage of the tree is transparency.

\paragraph{Why attendance and participation do not matter here.} Correlations of about @@CORRF_A@@ and @@CORRF_P@@ with the outcome are indistinguishable from zero even with one million students. This contradicts common classroom experience, and educational research generally reports an association between attendance and achievement. Together with the observations that @@SHARE100@@ of all total scores are exactly 100 and that the three predictors are mutually uncorrelated, this strongly suggests that the dataset is simulated, with the total score generated mainly from study hours plus random noise. The finding should be reported as a property of this dataset and not as evidence that attendance is unimportant in real classrooms.

\paragraph{Computational cost.} The entire pipeline, including @@NFITS@@ cross-validation fits, the depth sweep, the learning curve, permutation importance and all figures, runs in about @@RUNTIME@@ seconds on a laptop-class computer, and a prediction needs at most @@DEPTH@@ comparisons.

\paragraph{Limitations and assumptions.} (1) Probable synthetic data and only three predictors; no prior grades, no demographic variables, no timestamps, so a chronological split and grouped splits are not possible and the i.i.d.\ assumption is untested. (2) The positive class (D or F) is our choice; other cutoffs would change the results. (3) The error costs are illustrative. (4) Tuning used a $@@NTUNE@@$-student subsample for speed. (5) The final model is fitted on the training set only, so that the validation-based thresholds remain valid. (6) Performance is moderate, so the tool should start a conversation, not make a decision.

\paragraph{Ethical and social considerations.} Predictive tools in education carry risks. Labeling a student ``at-risk'' can create self-fulfilling expectations, and with precision near one half roughly half of the flagged students would be false alarms. Study-habit and attendance data may disadvantage students who work part-time or have caregiving duties, even when these factors are not recorded; before any use, the model should be audited for differences in error rates across student groups and monitored for dataset shift (Sec.~15.3). Student data must be handled in accordance with the Data Privacy Act of 2012 (Republic Act No.~10173) \cite{ra10173}. We recommend that predictions be visible only to the teacher and the student, used to offer support rather than penalties, always reviewed by a human, and accompanied by an explanation, which the readable tree rules provide.

\section{Conclusion}

This paper built and evaluated an interpretable Decision Tree Classifier that predicts, from behavioral data, whether a student will end with a D or F. Following the course workflow---a stratified training/validation/test split, leakage control, cross-validated tuning, validation-based thresholds, simple baselines, and a single test evaluation---a depth-@@DEPTH@@ tree with @@LEAVES@@ leaves reached ROC-AUC @@ROC@@ on @@NTEST@@ unseen students. At the $F_1$-optimal threshold it achieved precision @@PREC@@ and recall @@REC@@; at the cost-optimal threshold recall was @@REC_C@@ and the mistake cost per student was @@COST_TC@@ units, versus @@COST_NONE@@ for flagging no one. It matched a logistic-regression baseline while remaining readable as rules. Weekly self-study hours was the only informative predictor, whereas attendance and participation carried no signal, a result most plausibly explained by simulated data. The study shows that the choice of threshold and metric should follow the consequences of errors, not only accuracy.

\section{Recommendations and Future Directions}

\begin{enumerate}[label=\arabic*.]
    \item Validate the approach on real institutional data that include prior grades, term assessments, and demographic variables, with a fairness audit across student groups.
    \item Use a chronological split and group-aware validation when timestamps and class identifiers are available.
    \item Estimate real error costs with teachers and counselors and re-derive the threshold with Equation~\ref{eq:costthr}.
    \item Predict the numerical score with a regression tree and compare MAE, RMSE, and $R^2$ with a training-mean baseline, as in the regression portion of the lecture notes.
    \item Compare with ensembles and regularized models, and study calibration and monitoring after a pilot deployment.
\end{enumerate}

\newpage
\begin{thebibliography}{99}

\bibitem{baker2009}
Baker, R. S. J. d., \& Yacef, K. (2009). The state of educational data mining in 2009: A review and future visions. \textit{Journal of Educational Data Mining}, \textit{1}(1), 3--17. \url{https://jedm.educationaldatamining.org/index.php/JEDM/article/view/8}

\bibitem{breiman1984}
Breiman, L., Friedman, J. H., Olshen, R. A., \& Stone, C. J. (1984). \textit{Classification and regression trees}. Chapman and Hall/CRC. \url{https://doi.org/10.1201/9781315139470}

\bibitem{cortez2008}
Cortez, P., \& Silva, A. (2008). Using data mining to predict secondary school student performance. In A. Brito \& J. Teixeira (Eds.), \textit{Proceedings of the 5th FUture BUsiness TEChnology Conference} (pp. 5--12). EUROSIS. \url{https://hdl.handle.net/1822/8024}

\bibitem{kaufman2012}
Kaufman, S., Rosset, S., Perlich, C., \& Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. \textit{ACM Transactions on Knowledge Discovery from Data}, \textit{6}(4), Article 15. \url{https://doi.org/10.1145/2382577.2382579}

\bibitem{mitchell1997}
Mitchell, T. M. (1997). \textit{Machine learning}. McGraw-Hill. \url{http://www.cs.cmu.edu/~tom/mlbook.html}

\bibitem{quinlan1986}
Quinlan, J. R. (1986). Induction of decision trees. \textit{Machine Learning}, \textit{1}(1), 81--106. \url{https://doi.org/10.1007/BF00116251}

\bibitem{kaggle}
Qureshi, N. (n.d.). \textit{Student performance dataset} [Data set]. Kaggle. \url{https://www.kaggle.com/datasets/nabeelqureshitiii/student-performance-da}

\bibitem{ra10173}
Republic of the Philippines. (2012). \textit{Data Privacy Act of 2012} (Republic Act No. 10173). \url{https://www.privacy.gov.ph/data-privacy-act/}

\bibitem{romero2010}
Romero, C., \& Ventura, S. (2010). Educational data mining: A review of the state of the art. \textit{IEEE Transactions on Systems, Man, and Cybernetics, Part C}, \textit{40}(6), 601--618. \url{https://ieeexplore.ieee.org/document/5524021}

\bibitem{russell2022}
Russell, S. J., \& Norvig, P. (2022). \textit{Artificial intelligence: A modern approach} (4th ed., Global ed.). Pearson. \url{https://aima.cs.berkeley.edu/}

\bibitem{saito2015}
Saito, T., \& Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. \textit{PLOS ONE}, \textit{10}(3), Article e0118432. \url{https://doi.org/10.1371/journal.pone.0118432}

\end{thebibliography}

\newpage
\appendix
\section{Complete Rule Set of the Final Tree}\label{app:rules}

Table~\ref{tab:rules} lists every leaf of the final tree with its training size, estimated failure probability $\hat p$, and whether it triggers a flag at each of the two thresholds ($t_{F_1}=@@TF1@@$ and $t_{\mathrm{cost}}=@@TCOST@@$).

\begingroup\small
@@RULES_TABLE@@
\endgroup

\section{Reproducing the Results}

All code, data, figures, and result files are in the Git repository. After installing the requirements, the commands

\begin{lstlisting}[language=bash]
pip install -r requirements.txt
python -m tests.test_lesson_examples
python main.py
python paper/build_paper.py
\end{lstlisting}

run the unit tests against the lecture-note examples, regenerate all figures, results, and the trained model, and rebuild this paper, reproducing every number and figure in it. The random seed is @@SEED@@ throughout.

\end{document}
"""
V_["SEL_GAP"] = f"{sel_gap:.4f}"
body = TEMPLATE
for k, v in {**TOK, **V_}.items():
    body = body.replace(f"@@{k}@@", str(v))
OUT.write_text(body)
print("wrote", OUT)
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", OUT.name], cwd=OUT.parent, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        raise SystemExit("pdflatex failed")
for ext in ["aux", "log", "out", "toc"]:
    (OUT.parent / f"mini_research_paper.{ext}").unlink(missing_ok=True)
print("compiled", OUT.with_suffix(".pdf"))
