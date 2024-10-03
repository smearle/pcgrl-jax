import argparse, re, os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from os import path
import argparse
import json


class FeedbackGenerator:
    def __init__(self, config: dict):
        self.skill_log_csv = config.get('skill_log_csv', 'sample_skill_log.txt')
        self.feedback_type = config.get('feedback_type', 'statistics')
        self.postfix = config.get('postfix', 'outer_0')
        self.shared_storage_path = config.get('shared_storage_path', '.')
        self.result_path = path.join(self.shared_storage_path, 'Feedback', self.postfix, self.feedback_type)

    def run(self):
        if not os.path.exists(os.path.join(self.result_path)):
            os.makedirs(os.path.join(self.result_path))
        if self.feedback_type == 'statistics':
            self.generate_statistic(self.skill_log_csv)
        elif self.feedback_type == 't-SNE':
            self.generate_tsne(self.skill_log_csv)

    def generate_statistic(self, skill_log_csv):
        df = pd.read_csv(skill_log_csv)
        df_sorted = df[df['Step'] >= 20]
        pattern1 = r'^State\.Agent[0-3]\.Skill0\.Attack\..*$'
        pattern2 = r'^State\.Agent[0-3]\.Property\..*$'
        pattern3 = r'Episode'
        pattern4 = r'Playtesting.WinRate'
        filtered_columns = [col for col in df_sorted.columns if
                            re.match(pattern1, col) or re.match(pattern2, col) or
                            re.match(pattern3, col) or re.match(pattern4, col)]
        filtered_df = df_sorted[filtered_columns]
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
        filtered_df = filtered_df.nlargest(100, 'Episode')
        filtered_df = filtered_df.drop(columns=['Episode'])
        stats_df = filtered_df.describe().loc[['mean', 'std']]
        stats_dict = stats_df.to_dict()
        json_filename = path.join(self.result_path, "statistics.json")
        with open(json_filename, 'w') as json_file:
            json.dump(stats_dict, json_file)
        return path.join(self.result_path, "statistics.json")

    def generate_tsne(self, skill_log_csv):
        df = pd.read_csv(skill_log_csv)
        df_sorted = df[df['Step'] >= 20]
        pattern1 = r'^State\.Agent[0-3]\.Skill0\.Attack\..*$'
        pattern2 = r'^State\.Agent[0-3]\.Property\..*$'
        pattern3 = r'Episode'
        filtered_columns = [col for col in df_sorted.columns if
                            re.match(pattern1, col) or re.match(pattern2, col) or re.match(pattern3, col)]
        filtered_df = df_sorted[filtered_columns]
        filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()
        filtered_df = filtered_df.nlargest(100, 'Episode')
        filtered_df = filtered_df.drop(columns=['Episode'])
        agent_columns = {
            f'Agent{i}': [
                f'State.Agent{i}.Property.Health.Max',
                f'State.Agent{i}.Property.Armor',
                f'State.Agent{i}.Property.MoveSpeed',
                f'State.Agent{i}.Skill0.Attack.Range',
                f'State.Agent{i}.Skill0.Attack.Cooltime',
                f'State.Agent{i}.Skill0.Attack.Casttime',
                f'State.Agent{i}.Skill0.Attack.Amount'
            ] for i in range(4)
        }

        data_agents = []
        for i in range(4):
            agent_data = filtered_df[agent_columns[f'Agent{i}']].copy()
            agent_data.columns = [col.replace(f'Agent{i}', 'Agent') for col in agent_data.columns]
            agent_data['AgentNumber'] = i
            data_agents.append(agent_data)

        # 모든 에이전트의 데이터를 하나의 배열로 병합
        all_data = pd.concat(data_agents, axis=0, ignore_index=True)

        # t-SNE 적용 (2차원으로 축소)
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(all_data.drop(columns=['AgentNumber']))

        # 결과를 데이터프레임으로 변환
        tsne_df = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
        tsne_df['AgentNumber'] = all_data['AgentNumber']

        # 그래프 그리기
        plt.figure(figsize=(10, 8))
        colors = ['r', 'g', 'b', 'c']

        for i, color in enumerate(colors):
            plt.scatter(tsne_df[tsne_df['AgentNumber'] == i]['tsne1'], tsne_df[tsne_df['AgentNumber'] == i]['tsne2'],
                        c=color, label=f'Agent{i}', alpha=0.7)

        plt.legend()
        plt.title('t-SNE visualization of Agent Parameters')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True)

        plt.savefig(path.join(self.result_path, "t-SNE.png"))
        return path.join(self.result_path, "t-SNE.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill_log_csv', type=str, default='sample_skill_log.txt')
    parser.add_argument('--feedback_type', type=str, default='statistics') # [t-SNE, statistics]
    parser.add_argument('--postfix', type=str, default='outer_0')
    parser.add_argument('--shared_storage_path', type=str, default='.')
    args = parser.parse_args()

    args = vars(args)
    feedback_generator = FeedbackGenerator(args)
    feedback_generator.run()