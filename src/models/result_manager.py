import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


class ResultManager():
    def __init__(self, data_path: Path, plot_path: Path,file_name_pattern: str):
        self.__df = self.__get_files(data_path, file_name_pattern)
        self.__plot_path = plot_path

    def __get_files(self, path: Path, file_name_pattern: str)-> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for file in sorted(path.glob(file_name_pattern)):
            generation_number = int(file.stem.split('_')[2])
            with open(file) as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df['generation'] = generation_number
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)
    
    def mean_fitness(self)-> None:
        plt.figure(figsize=(10, 5))
        self.__df.groupby("generation")["fitness"].mean().plot(marker='o')
        plt.title("Fitness médio por geração")
        plt.xlabel("Geração")
        plt.ylabel("Fitness")
        plt.grid(True)
        plt.savefig(self.__plot_path / 'mean_fitness.png', dpi=300)
        plt.close()

    def distance_vs_kill(self)-> None:
        plt.figure(figsize=(8, 6))
        plt.scatter(self.__df["distance"], self.__df["kill_amount"], c=self.__df["generation"], cmap="viridis", alpha=0.7)
        plt.title("Kill Amount vs Distance")
        plt.xlabel("Distância Percorrida")
        plt.ylabel("Inimigos Eliminados")
        plt.colorbar(label="Geração")
        plt.tight_layout()
        plt.savefig(self.__plot_path / 'distance_vs_kill.png', dpi=300)
        plt.close()

    def secondary_mean_evolutuion(self)-> None:
        agg = self.__df.groupby("generation")[["kill_amount", "distance", "healt"]].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(agg.index, agg["kill_amount"], label="Kills")
        plt.plot(agg.index, agg["distance"], label="Distância")
        plt.plot(agg.index, agg["healt"], label="Health")
        plt.title("Média de comportamento por geração")
        plt.xlabel("Geração")
        plt.ylabel("Valor médio")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.__plot_path / 'secondary_mean_evolutuion.png', dpi=300)
        plt.close()
