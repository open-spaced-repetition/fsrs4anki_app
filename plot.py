from tqdm.auto import trange
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px


def make_plot(proj_dir, type_sequence, w, difficulty_distribution_padding, progress=gr.Progress(track_tqdm=True)):
    base = 1.01
    index_len = 800
    index_offset = 150
    d_range = 10
    d_offset = 1
    r_repetitions = 1
    f_repetitions = 2.3
    max_repetitions = 200000

    type_block = dict()
    type_count = dict()
    last_t = type_sequence[0]
    type_block[last_t] = 1
    type_count[last_t] = 1
    for t in type_sequence[1:]:
        type_count[t] = type_count.setdefault(t, 0) + 1
        if t != last_t:
            type_block[t] = type_block.setdefault(t, 0) + 1
        last_t = t
    if 2 in type_count and 2 in type_block:
        f_repetitions = round(type_count[2] / type_block[2] + 1, 1)

    def stability2index(stability):
        return int(round(np.log(stability) / np.log(base)) + index_offset)

    def init_stability(d):
        return max(((d - w[2]) / w[3] + 2) * w[1] + w[0], np.power(base, -index_offset))

    def cal_next_recall_stability(s, r, d, response):
        if response == 1:
            return s * (1 + np.exp(w[6]) * (11 - d) * np.power(s, w[7]) * (np.exp((1 - r) * w[8]) - 1))
        else:
            return w[9] * np.power(d, w[10]) * np.power(s, w[11]) * np.exp((1 - r) * w[12])

    stability_list = np.array([np.power(base, i - index_offset) for i in range(index_len)])
    # print(f"terminal stability: {stability_list.max(): .2f}")
    df = pd.DataFrame(columns=["retention", "difficulty", "repetitions"])

    for percentage in trange(96, 70, -2, desc='Repetition vs Retention plot'):
        recall = percentage / 100
        repetitions_list = np.zeros((d_range, index_len))
        repetitions_list[:, :-1] = max_repetitions
        for d in range(d_range, 0, -1):
            s0 = init_stability(d)
            s0_index = stability2index(s0)
            diff = max_repetitions
            while diff > 0.1:
                s0_repetitions = repetitions_list[d - 1][s0_index]
                for s_index in range(index_len - 2, -1, -1):
                    stability = stability_list[s_index];
                    interval = max(1, round(stability * np.log(recall) / np.log(0.9)))
                    p_recall = np.power(0.9, interval / stability)
                    recall_s = cal_next_recall_stability(stability, p_recall, d, 1)
                    forget_d = min(d + d_offset, 10)
                    forget_s = cal_next_recall_stability(stability, p_recall, forget_d, 0)
                    recall_s_index = min(stability2index(recall_s), index_len - 1)
                    forget_s_index = min(max(stability2index(forget_s), 0), index_len - 1)
                    recall_repetitions = repetitions_list[d - 1][recall_s_index] + r_repetitions
                    forget_repetitions = repetitions_list[forget_d - 1][forget_s_index] + f_repetitions
                    exp_repetitions = p_recall * recall_repetitions + (1.0 - p_recall) * forget_repetitions
                    if exp_repetitions < repetitions_list[d - 1][s_index]:
                        repetitions_list[d - 1][s_index] = exp_repetitions
                diff = s0_repetitions - repetitions_list[d - 1][s0_index]
            df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [recall, d, s0_repetitions]

    df.sort_values(by=["difficulty", "retention"], inplace=True)
    df.to_csv(proj_dir/"expected_repetitions.csv", index=False)
    # print("expected_repetitions.csv saved.")

    optimal_retention_list = np.zeros(10)
    df2 = pd.DataFrame()
    for d in range(1, d_range + 1):
        retention = df[df["difficulty"] == d]["retention"]
        repetitions = df[df["difficulty"] == d]["repetitions"]
        optimal_retention = retention.iat[repetitions.argmin()]
        optimal_retention_list[d - 1] = optimal_retention
        df2 = df2.append(
            pd.DataFrame({'retention': retention, 'expected repetitions': repetitions, 'd': d, 'r': optimal_retention}))

    fig = px.line(df2, x="retention", y="expected repetitions", color='d', log_y=True)

    # print(f"\n-----suggested retention: {np.inner(difficulty_distribution_padding, optimal_retention_list):.2f}-----")
    suggested_retention_markdown = f"""# Suggested Retention: `{np.inner(difficulty_distribution_padding, optimal_retention_list):.2f}`"""
    return fig, suggested_retention_markdown
