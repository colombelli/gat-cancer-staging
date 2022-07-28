import shutil

from_base = "C:/Users/colombelli/Desktop/TCC/experiments/"
to_base = "C:/Users/colombelli/Desktop/TCC/experiments_extra_20/"

for cancer in ["KIRC", "COAD", "LUAD"]:
    print(cancer)
    for strategy in ['correlation', 'correlation_multi_omics', 'snf']:
        print("\t", strategy)
        for th in ['09', '095', '099']:
            print("\t\t", th)
            from_path = f"{from_base}{cancer}/{strategy}/{th}/"
            to_path = f"{to_base}{cancer}/{strategy}/{th}/"

            shutil.copytree(from_path+"hp_tuner_gat", to_path+"hp_tuner_gat")
            shutil.copytree(from_path+"hp_tuner_gcn", to_path+"hp_tuner_gcn")

            files = ["edges.csv", "gat_results.csv", "gcn_results.csv", "mlp_results.csv"]
            for f in files:
                shutil.copyfile(from_path+f, to_path+f)