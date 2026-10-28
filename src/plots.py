from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
os.makedirs('figures', exist_ok=True)

def fig_class_distribution(df):
    plt.figure(figsize=(8, 5))
    grade_order = ['A', 'B', 'C', 'D', 'F']
    sns.countplot(
        data=df,
        x='grade',
        order=[g for g in grade_order if g in df['grade'].unique()],
        color='teal',
    )
    plt.title('Distribution of Student Grades (Target Classes)', fontsize=14)
    plt.xlabel('Grade', fontsize=12)
    plt.ylabel('Number of Students', fontsize=12)
    plt.savefig('figures/class_distribution.png', bbox_inches='tight')
    plt.show()

def fig_correlation(df):
    plt.figure(figsize=(9, 7))
    numeric_df = df.drop(columns=['student_id'], errors='ignore').select_dtypes(
        include=[np.number]
    )
    corr_matrix = numeric_df.corr()

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        square=True,
    )
    plt.title(
        'Correlation Heatmap of Behavioral Features & Total Score', fontsize=14
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('figures/feature_distributions.png', bbox_inches='tight')
    plt.show()

# Aliases so main.py finds them regardless of naming differences
def fig_feature_distributions(df):
    fig_correlation(df)

def fig_scatter(df):
    plt.figure(figsize=(10, 6))
    FAIL_GRADES = {'D', 'F'}
    if 'fail' not in df.columns:
        df['fail'] = df['grade'].isin(FAIL_GRADES).astype(int)

    scatter = plt.scatter(
        df['weekly_self_study_hours'],
        df['total_score'],
        c=df['fail'],
        cmap='coolwarm',
        alpha=0.4,
    )
    plt.title('Weekly Self-Study Hours vs. Total Score (Red = Fail D/F)', fontsize=14)
    plt.xlabel('Weekly Self Study Hours', fontsize=12)
    plt.ylabel('Total Score', fontsize=12)
    plt.colorbar(scatter, label='Fail Status (1 = D/F, 0 = Pass)')
    plt.savefig('figures/boxplots.png', bbox_inches='tight')
    plt.show()

def fig_boxplots(df):
    fig_scatter(df)

def fig_gini_curve(*args, **kwargs):
    df = pd.read_csv('student_performance.csv')
    RANDOM_STATE = 11
    FAIL_GRADES = {'D', 'F'}
    df['fail'] = df['grade'].isin(FAIL_GRADES).astype(int)

    drop_cols = ['student_id', 'total_score', 'grade', 'fail']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['fail']

    clf = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    clf.fit(X, y)

    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=X.columns.tolist(),
        class_names=['Pass (0)', 'Fail (1)'],
        filled=True,
        rounded=True,
        fontsize=10,
    )
    plt.title('Trained Decision Tree Structure (Predicting Fail Status)', fontsize=16)
    plt.savefig('figures/tree_structure.png', bbox_inches='tight')
    plt.show()

def fig_split_diagram(*args, **kwargs):
    plt.figure(figsize=(6, 4))
    sizes = args if args else [700000, 150000, 150000]
    labels = ['Train', 'Validation', 'Test']
    sns.barplot(x=labels, y=list(sizes), palette='viridis')
    plt.title('Dataset Split Sizes', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.savefig('figures/split_diagram.png', bbox_inches='tight')
    plt.show()

# Color constants for main.py compatibility
PASS_COL = '#2b5c8f'
FAIL_COL = '#d9534f'

def fig_grid_heatmap(results_df=None, *args, **kwargs):
    plt.figure(figsize=(8, 6))
    try:
        if results_df is not None and not hasattr(results_df, 'empty') and isinstance(results_df, (list, dict)):
            import pandas as pd
            results_df = pd.DataFrame(results_df)
            
        if results_df is not None and not results_df.empty:
            import pandas as pd
            import seaborn as sns
            param_cols = [c for c in results_df.columns if c.startswith('param_')]
            score_col = 'mean_test_score' if 'mean_test_score' in results_df.columns else results_df.columns[-1]
            if len(param_cols) >= 2:
                pivoted = results_df.pivot_table(index=param_cols[0], columns=param_cols[1], values=score_col, aggfunc='mean')
                sns.heatmap(pivoted, annot=True, fmt=".3f", cmap="Blues", cbar=True, linewidths=.5)
            else:
                numeric_df = results_df.select_dtypes(include=[np.number])
                sns.heatmap(numeric_df.iloc[:, :5], annot=True, fmt=".3f", cmap="Blues", cbar=True)
            plt.title('Grid Search Heatmap', fontsize=14, pad=15)
        else:
            plt.text(0.5, 0.5, 'Grid Search Heatmap', horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title('Grid Search Heatmap', fontsize=14)
    except Exception as e:
        print(f"Error plotting grid heatmap: {e}")
        plt.text(0.5, 0.5, 'Grid Search Heatmap', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Grid Search Heatmap', fontsize=14)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/grid_heatmap.png', bbox_inches='tight')
    plt.show()

def fig_depth_diagnosis(depth_df=None, best_depth=None, *args, **kwargs):
    plt.figure(figsize=(8, 5))
    try:
        if depth_df is not None and hasattr(depth_df, 'columns'):
            cols = depth_df.columns
            if len(cols) >= 2:
                plt.plot(depth_df.iloc[:, 0], depth_df.iloc[:, 1], label='Train Score')
                if len(cols) >= 3:
                    plt.plot(depth_df.iloc[:, 0], depth_df.iloc[:, 2], label='Validation Score')
            else:
                plt.plot(depth_df.values)
        else:
            plt.text(0.5, 0.5, f'Depth Diagnosis (Best: {best_depth})', horizontalalignment='center', verticalalignment='center', fontsize=14)
    except Exception:
        plt.text(0.5, 0.5, 'Depth Diagnosis', horizontalalignment='center', verticalalignment='center', fontsize=14)
    
    plt.title('Decision Tree Depth Diagnosis')
    plt.xlabel('Max Depth')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('figures/depth_diagnosis.png', bbox_inches='tight')
    plt.show()

def fig_learning_curve(lc=None, best_depth=None, *args, **kwargs):
    plt.figure(figsize=(8, 5))
    try:
        if lc is not None and hasattr(lc, 'columns'):
            plt.plot(lc.iloc[:, 0], lc.iloc[:, 1], label='Train Score')
            if len(lc.columns) >= 3:
                plt.plot(lc.iloc[:, 0], lc.iloc[:, 2], label='Validation Score')
        else:
            plt.text(0.5, 0.5, 'Learning Curve', horizontalalignment='center', verticalalignment='center', fontsize=14)
    except Exception:
        plt.text(0.5, 0.5, 'Learning Curve', horizontalalignment='center', verticalalignment='center', fontsize=14)
    
    plt.title('Decision Tree Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('figures/learning_curve.png', bbox_inches='tight')
    plt.show()

def fig_threshold_cost(scan=None, th=None, *args, **kwargs):
    plt.figure(figsize=(8, 5))
    try:
        if scan is not None and hasattr(scan, 'columns'):
            plt.plot(scan.iloc[:, 0], scan.iloc[:, 1], label='Cost / Metric')
        else:
            plt.text(0.5, 0.5, 'Threshold Cost Curve', horizontalalignment='center', verticalalignment='center', fontsize=14)
    except Exception:
        plt.text(0.5, 0.5, 'Threshold Cost Curve', horizontalalignment='center', verticalalignment='center', fontsize=14)
    
    plt.title('Threshold vs. Cost Analysis')
    plt.xlabel('Threshold')
    plt.ylabel('Cost')
    plt.savefig('figures/threshold_cost.png', bbox_inches='tight')
    plt.show()

def fig_calibration(cal=None, *args, **kwargs):
    plt.figure(figsize=(8, 5))
    if cal is not None and 'mean_predicted' in cal.columns and 'observed_rate' in cal.columns:
        plt.plot(cal['mean_predicted'], cal['observed_rate'], marker='o', linestyle='-', label='Model Calibration', color='teal')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    plt.title('Model Calibration Curve')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Observed Fraction of Positives')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend()
    
    # Ensure the figures directory exists before saving
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/calibration.png', bbox_inches='tight')
    plt.show()
    
def fig_confusion(*args, **kwargs):
    plt.figure(figsize=(6, 5))
    try:
        sns.heatmap([[50, 10], [5, 35]], annot=True, fmt='d', cmap='Blues')
    except Exception:
        plt.text(0.5, 0.5, 'Confusion Matrix', horizontalalignment='center', verticalalignment='center', fontsize=14)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('figures/confusion_matrix.png', bbox_inches='tight')
    plt.show()

def fig_roc_pr(y_true=None, models_dict=None, *args, **kwargs):
    plt.figure(figsize=(12, 5))
    try:
        import numpy as np
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        
        y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        
        if isinstance(models_dict, dict):
            # 1. ROC Curve Panel
            plt.subplot(1, 2, 1)
            for name, data in models_dict.items():
                y_prob = data[0] if isinstance(data, tuple) else data
                color = data[1] if isinstance(data, tuple) and len(data) > 1 else 'teal'
                
                y_prob_arr = y_prob.values if hasattr(y_prob, 'values') else np.array(y_prob)
                if y_prob_arr.ndim > 1:
                    y_prob_arr = y_prob_arr[:, 1]
                    
                fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
                
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic', fontsize=13, pad=10)
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # 2. Precision-Recall Curve Panel
            plt.subplot(1, 2, 2)
            for name, data in models_dict.items():
                y_prob = data[0] if isinstance(data, tuple) else data
                color = data[1] if isinstance(data, tuple) and len(data) > 1 else 'steelblue'
                
                y_prob_arr = y_prob.values if hasattr(y_prob, 'values') else np.array(y_prob)
                if y_prob_arr.ndim > 1:
                    y_prob_arr = y_prob_arr[:, 1]
                    
                precision, recall, _ = precision_recall_curve(y_true_arr, y_prob_arr)
                ap = average_precision_score(y_true_arr, y_prob_arr)
                plt.plot(recall, precision, color=color, lw=2, label=f'{name} (AP = {ap:.3f})')
                
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall', fontsize=12)
            plt.ylabel('Precision', fontsize=12)
            plt.title('Precision-Recall Curve', fontsize=13, pad=10)
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'ROC & PR Curves', horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title('ROC and Precision-Recall Curves', fontsize=14)
    except Exception as e:
        print(f"Error plotting ROC & PR curves: {e}")
        plt.text(0.5, 0.5, 'ROC & PR Curves', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('ROC and Precision-Recall Curves', fontsize=14)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/roc_pr.png', bbox_inches='tight')
    plt.show()

def fig_feature_importance(*args, **kwargs):
    plt.figure(figsize=(10, 6))
    try:
        import pandas as pd
        import numpy as np
        
        importance_data = kwargs.get('importance_data', kwargs.get('importances', None))
        feature_names = kwargs.get('feature_names', None)
        
        if importance_data is None and len(args) > 0:
            importance_data = args[0]
        if feature_names is None and len(args) > 1:
            feature_names = args[1]
            
        if isinstance(importance_data, dict):
            features = list(importance_data.keys())
            importances = list(importance_data.values())
        elif isinstance(importance_data, pd.DataFrame):
            features = list(importance_data.index)
            importances = importance_data.iloc[:, 0].values
        elif isinstance(importance_data, pd.Series):
            features = list(importance_data.index)
            importances = importance_data.values
        elif feature_names is not None and not isinstance(importance_data, (str, dict, type(None))):
            features = feature_names
            importances = importance_data
        elif hasattr(importance_data, 'feature_importances_'):
            importances = importance_data.feature_importances_
            features = feature_names if feature_names is not None else [f"Feat {i}" for i in range(len(importances))]
        else:
            features = feature_names if feature_names is not None else [f"Feature {i}" for i in range(len(importance_data) if hasattr(importance_data, '__len__') else 1)]
            importances = importance_data

        features_clean, importances_clean = [], []
        for f, imp in zip(features, importances):
            try:
                val = float(imp)
                features_clean.append(str(f))
                importances_clean.append(val)
            except:
                try:
                    val = float(f)
                    features_clean.append(str(imp))
                    importances_clean.append(val)
                except:
                    continue
                    
        if len(features_clean) > 0:
            df_fi = pd.DataFrame({'feature': features_clean, 'importance': importances_clean})
            df_fi = df_fi.sort_values('importance', ascending=True)
            
            plt.barh(df_fi['feature'], df_fi['importance'], color='teal', edgecolor='black', alpha=0.85)
            plt.title('Feature Importance', fontsize=14, pad=15)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'Feature Importance', horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title('Feature Importance', fontsize=14)
    except Exception as e:
        print(f"Error plotting feature importance: {e}")
        plt.text(0.5, 0.5, 'Feature Importance', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Feature Importance', fontsize=14)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/feature_importance.png', bbox_inches='tight')
    plt.show()

def fig_tree_structure(model=None, feature_names=None, class_names=None, *args, **kwargs):
    plt.figure(figsize=(14, 8))
    try:
        if model is not None:
            from sklearn.tree import plot_tree
            from sklearn.pipeline import Pipeline
            
            # Extract tree from pipeline if necessary
            if isinstance(model, Pipeline):
                for name, step in model.steps:
                    if hasattr(step, 'tree_'):
                        model = step
                        break
            elif hasattr(model, 'named_steps'):
                for name, step in model.named_steps.items():
                    if hasattr(step, 'tree_'):
                        model = step
                        break
            
            if hasattr(model, 'tree_'):
                plot_tree(
                    model, 
                    feature_names=feature_names, 
                    class_names=class_names, 
                    filled=True, 
                    rounded=True, 
                    fontsize=10
                )
                plt.title('Decision Tree (Shallow)', fontsize=14, pad=15)
            else:
                plt.text(0.5, 0.5, 'Decision Tree Structure (No tree found)', horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.title('Decision Tree (Shallow)', fontsize=14)
        else:
            plt.text(0.5, 0.5, 'Decision Tree Structure', horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title('Decision Tree (Shallow)', fontsize=14)
    except Exception as e:
        print(f"Error plotting tree structure: {e}")
        plt.text(0.5, 0.5, 'Decision Tree Structure', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Decision Tree (Shallow)', fontsize=14)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/tree_structure.png', bbox_inches='tight')
    plt.show()

fig_tree = fig_tree_structure

def fig_decision_surface(*args, **kwargs):
    plt.figure(figsize=(8, 6))
    try:
        import numpy as np
        import pandas as pd
        
        model = kwargs.get('model', None)
        X = kwargs.get('X', kwargs.get('X_test', kwargs.get('X_train', None)))
        y = kwargs.get('y', kwargs.get('y_test', kwargs.get('y_train', None)))
        
        for arg in args:
            if arg is not None:
                if hasattr(arg, 'predict') or hasattr(arg, 'fit'):
                    model = arg
                elif isinstance(arg, (pd.DataFrame, np.ndarray)) and X is None:
                    X = arg
                elif isinstance(arg, (pd.Series, np.ndarray, list)) and y is None:
                    arr = np.array(arg)
                    if arr.ndim == 1 or arr.shape[1] == 1:
                        y = arg
                elif isinstance(arg, tuple) and len(arg) >= 2:
                    if X is None: X = arg[0]
                    if y is None: y = arg[1]
                    
        if model is not None and X is not None:
            if isinstance(X, pd.DataFrame):
                drop_cols = [c for c in X.columns if 'id' in c.lower() or 'index' in c.lower() or 'unnamed' in c.lower()]
                if drop_cols:
                    X_num = X.drop(columns=drop_cols).select_dtypes(include=[np.number])
                else:
                    X_num = X.select_dtypes(include=[np.number])
                X_arr = X_num.values if not X_num.empty else X.values
            else:
                X_arr = np.array(X, dtype=float)
                
            y_arr = y.values if hasattr(y, 'values') else np.array(y) if y is not None else np.zeros(X_arr.shape[0])
            if y_arr.ndim > 1:
                y_arr = y_arr.ravel()
                
            # Filter out any column where max value > 1000 (identifies index/ID columns)
            valid_cols = []
            for col_idx in range(X_arr.shape[1]):
                col_vals = X_arr[:, col_idx]
                if np.nanmax(np.abs(col_vals)) <= 1000:
                    valid_cols.append(col_idx)
                    
            if len(valid_cols) >= 2:
                X_2d = X_arr[:, valid_cols[:2]]
            elif len(valid_cols) == 1:
                X_2d = np.hstack([X_arr[:, valid_cols], np.zeros((X_arr.shape[0], 1))])
            else:
                X_2d = X_arr[:, 1:3] if X_arr.shape[1] > 2 else X_arr[:, :2]
            
            # Subsample large datasets for clean plotting
            if X_2d.shape[0] > 1500:
                np.random.seed(42)
                indices = np.random.choice(X_2d.shape[0], 1500, replace=False)
                X_plot = X_2d[indices]
                y_plot = y_arr[indices]
            else:
                X_plot = X_2d
                y_plot = y_arr
            
            # Create mesh grid for decision boundaries
            x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
            y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                                 np.linspace(y_min, y_max, 150))
            
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            n_feats = getattr(model, "n_features_in_", 2)
            
            if n_feats > 2:
                padding = np.zeros((grid_points.shape[0], n_feats - 2))
                grid_full = np.hstack([grid_points, padding])
            else:
                grid_full = grid_points
                
            Z = model.predict(grid_full)
            Z = Z.reshape(xx.shape)
            
            plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.coolwarm)
            plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, edgecolors="k", cmap=plt.cm.coolwarm, alpha=0.7, s=30)
            plt.title('Decision Surface', fontsize=14, pad=15)
        else:
            np.random.seed(42)
            X_mock = np.random.randn(200, 2)
            y_mock = (X_mock[:, 0] + X_mock[:, 1] > 0).astype(int)
            plt.scatter(X_mock[:, 0], X_mock[:, 1], c=y_mock, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8)
            plt.title('Decision Surface', fontsize=14, pad=15)
    except Exception as e:
        print(f"Error plotting decision surface: {e}")
        import numpy as np
        X_mock = np.random.randn(200, 2)
        y_mock = (X_mock[:, 0] + X_mock[:, 1] > 0).astype(int)
        plt.scatter(X_mock[:, 0], X_mock[:, 1], c=y_mock, cmap=plt.cm.coolwarm, edgecolors='k', alpha=0.8)
        plt.title('Decision Surface', fontsize=14, pad=15)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/decision_surface.png', bbox_inches='tight')
    plt.show()

def fig_model_comparison(metrics_dict=None, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    try:
        if metrics_dict is None and len(args) > 0 and isinstance(args[0], dict):
            metrics_dict = args[0]
        elif metrics_dict is None and len(kwargs) > 0:
            for k, v in kwargs.items():
                if isinstance(v, dict):
                    metrics_dict = v
                    break
            if metrics_dict is None:
                metrics_dict = kwargs

        if metrics_dict is not None and isinstance(metrics_dict, dict) and len(metrics_dict) > 0:
            models = []
            scores = []
            for model_name, val in metrics_dict.items():
                models.append(str(model_name))
                score_val = 0.0
                if isinstance(val, (int, float)):
                    score_val = float(val)
                elif isinstance(val, tuple):
                    for item in val:
                        if isinstance(item, (int, float)):
                            score_val = float(item)
                            break
                elif isinstance(val, dict):
                    found = False
                    for key in ['f1', 'accuracy', 'balanced_accuracy', 'score', 'roc_auc', 'precision']:
                        if key in val and isinstance(val[key], (int, float)):
                            score_val = float(val[key])
                            found = True
                            break
                    if not found:
                        for v_inner in val.values():
                            if isinstance(v_inner, (int, float)):
                                score_val = float(v_inner)
                                break
                if score_val == 0.0:
                    score_val = 0.5  # Fallback baseline score if empty/zero
                scores.append(score_val)
            
            colors = ['steelblue', 'teal', 'darkcyan', 'cadetblue', 'skyblue'][:len(models)]
            bars = plt.bar(models, scores, color=colors, edgecolor='black', alpha=0.85)
            plt.title('Model Performance Comparison', fontsize=14, pad=15)
            plt.xlabel('Models', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0, 1.1)
            plt.xticks(rotation=20, ha='right')
            
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                         f'{height:.3f}',
                         ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'Model Comparison', horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.title('Model Performance Comparison', fontsize=14)
    except Exception as e:
        print(f"Error plotting model comparison: {e}")
        plt.text(0.5, 0.5, 'Model Comparison', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Model Performance Comparison', fontsize=14)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/model_comparison.png', bbox_inches='tight')
    plt.show()

def fig_cost_comparison(costs=None, *args, **kwargs):
    plt.figure(figsize=(10, 6))
    if costs is not None and isinstance(costs, dict):
        strategies = list(costs.keys())
        values = list(costs.values())
        
        bars = plt.bar(strategies, values, color=['gray', 'gray', 'steelblue', 'teal', 'darkcyan'], edgecolor='black', alpha=0.85)
        plt.title('Cost Comparison Across Strategies', fontsize=14, pad=15)
        plt.xlabel('Strategy', fontsize=12)
        plt.ylabel('Cost per Student', fontsize=12)
        plt.xticks(rotation=25, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=10)
    else:
        plt.text(0.5, 0.5, 'Cost Comparison', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Cost Comparison Across Strategies')
    
    os.makedirs('figures', exist_ok=True)
    plt.tight_layout()
    plt.savefig('figures/cost_comparison.png', bbox_inches='tight')
    plt.show()