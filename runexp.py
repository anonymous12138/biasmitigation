from Baseline.Random import blind_random
from xFAIR import *
from Baseline.Reweighing import reweigh
from Baseline.FairSMOTE import Fair_Smote

if __name__ == "__main__":
    filenames = ['adult', 'compas-scores-two-years', 'GermanData', 'bank', 'heart', 'default', 'h181']
    keywords = {'adult': ['sex', 'race'],
                'compas-scores-two-years': ['sex', 'race'],
                'bank': ['age'],
                'default': ['sex'],
                'GermanData': ['sex'],
                'h181': ['race'],
                'heart': ['age']
                }
    base = RandomForestClassifier()
    base2 = DecisionTreeRegressor()
    for each in filenames:
        fname = each
        klist = keywords[fname]
        for keyword in klist:
            # Fair-SMOTE
            df1 = pd.read_csv("./Data/"+fname + "_processed.csv")
            result1 = Fair_Smote(df1, base, keyword=keyword, rep=10)
            a, p, r, f, ao, eo, spd, di,fr = result1
            print("**"*50)
            print("Fair-SMOTE",fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))
            print("-FR", np.mean(fr))

            # Reweighing
            df1 = pd.read_csv("./Data/" + fname + "_processed.csv")
            result1 = reweigh(base,df1, keyword=keyword, rep=10)
            a, p, r, f, ao, eo, spd, di, fr = result1
            print("**" * 50)
            print("Reweighing",fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))
            print("-FR", np.mean(fr))

            # Random
            df1 = pd.read_csv("./Data/" + fname + "_processed.csv")
            result1 = blind_random(base, df1, keyword=keyword, rep=10)
            a, p, r, f, ao, eo, spd, di, fr = result1
            print("**" * 50)
            print("Random",fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))
            print("-FR", np.mean(fr))

            # xFAIR
            df1 = pd.read_csv("./Data/" + fname + "_processed.csv")
            result1 = xFAIR(df1,base, base2, keyword=keyword, rep=10)
            a, p, r, f, ao, eo, spd, di, fr = result1
            print("**" * 50)
            print("xFAIR",fname, keyword)
            print("+Accuracy", np.mean(a))
            print("+Precision", np.mean(p))
            print("+Recall", np.mean(r))
            print("+F1", np.mean(f))
            print("-AOD", np.mean(ao))
            print("-EOD", np.mean(eo))
            print("-SPD", np.mean(spd))
            print("-DI", np.mean(di))