import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

# --- 1. load data ---
df = pd.read_csv("./data/all_data.csv")
X_cols = ['PB','PB@Ni(OH)2','PB@Ni-MOF']
ions = ['Pb','Cd','Cu','Fe','K']

models_cls = {
    'RF (Balanced)': RandomForestClassifier(random_state=42, class_weight='balanced'),
    # 'MLP (Default)': MLPClassifier(max_iter=1000, random_state=42) # abandoned
}


# --- 2. set cross-validation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'Accuracy': 'accuracy',
    'Precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'Recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'F1_Score': make_scorer(f1_score, average='weighted', zero_division=0)
}

# --- 3. def evaluate fuct ---
def evaluate_dataset(df):
    
    for ion in ions:
        y = (df[ion] > 0).astype(int)

        counts = y.value_counts()
        print(f"\n-- {ion} (存在性) 评估 -- | 标签分布: 1={counts.get(1, 0)}, 0={counts.get(0, 0)}")

        if len(counts) < 2:
             print(f"警告: {ion} 标签只有一类，跳过交叉验证.")
             continue
        
        X = df[X_cols]
        
        for model_name, model in models_cls.items():
            results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            
            acc_mean = results['test_Accuracy'].mean()
            prec_mean = results['test_Precision'].mean()
            rec_mean = results['test_Recall'].mean()
            f1_mean = results['test_F1_Score'].mean()
            
            print(f"  {model_name:<15} | Acc: {acc_mean:.3f} | Prec: {prec_mean:.3f} | Rec: {rec_mean:.3f} | F1: {f1_mean:.3f}")

if __name__ == "__main__":
    evaluate_dataset(df)