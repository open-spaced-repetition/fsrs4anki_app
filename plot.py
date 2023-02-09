from tqdm.auto import trange
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px


def make_plot(proj_dir, type_sequence, time_sequence, w, difficulty_distribution_padding, progress=gr.Progress(track_tqdm=True)):
    base = 1.01
    index_len = 793
    index_offset = 200
    d_range = 10
    d_offset = 1
    r_time = 8
    f_time = 25
    max_time = 200000

    type_block = dict()
    type_count = dict()
    type_time = dict()
    last_t = type_sequence[0]
    type_block[last_t] = 1
    type_count[last_t] = 1
    type_time[last_t] = time_sequence[0]
    for i,t in enumerate(type_sequence[1:]):
        type_count[t] = type_count.setdefault(t, 0) + 1
        type_time[t] = type_time.setdefault(t, 0) + time_sequence[i]
        if t != last_t:
            type_block[t] = type_block.setdefault(t, 0) + 1
        last_t = t

    r_time = round(type_time[1]/type_count[1]/1000, 1)

    if 2 in type_count and 2 in type_block:
        f_time = round(type_time[2]/type_block[2]/1000 + r_time, 1)

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
    df = pd.DataFrame(columns=["retention", "difficulty", "time"])

    for percentage in trange(96, 66, -2, desc='Time vs Retention plot'):
        recall = percentage / 100
        time_list = np.zeros((d_range, index_len))
        time_list[:,:-1] = max_time
        for d in range(d_range, 0, -1):
            s0 = init_stability(d)
            s0_index = stability2index(s0)
            diff = max_time
            while diff > 0.1:
                s0_time = time_list[d - 1][s0_index]
                for s_index in range(index_len - 2, -1, -1):
                    stability = stability_list[s_index];
                    interval = max(1, round(stability * np.log(recall) / np.log(0.9)))
                    p_recall = np.power(0.9, interval / stability)
                    recall_s = cal_next_recall_stability(stability, p_recall, d, 1)
                    forget_d = min(d + d_offset, 10)
                    forget_s = cal_next_recall_stability(stability, p_recall, forget_d, 0)
                    recall_s_index = min(stability2index(recall_s), index_len - 1)
                    forget_s_index = min(max(stability2index(forget_s), 0), index_len - 1)
                    recall_time = time_list[d - 1][recall_s_index] + r_time
                    forget_time = time_list[forget_d - 1][forget_s_index] + f_time
                    exp_time = p_recall * recall_time + (1.0 - p_recall) * forget_time
                    if exp_time < time_list[d - 1][s_index]:
                        time_list[d - 1][s_index] = exp_time
                diff = s0_time - time_list[d - 1][s0_index]
            df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [recall, d, s0_time]


    df.sort_values(by=["difficulty", "retention"], inplace=True)
    df.to_csv(proj_dir/"expected_time.csv", index=False)
    # print("expected_repetitions.csv saved.")

    optimal_retention_list = np.zeros(10)
    df2 = pd.DataFrame()
    for d in range(1, d_range + 1):
        retention = df[df["difficulty"] == d]["retention"]
        time = df[df["difficulty"] == d]["time"]
        optimal_retention = retention.iat[time.argmin()]
        optimal_retention_list[d - 1] = optimal_retention
        df2 = df2.append(
            pd.DataFrame({'retention': retention, 'expected time': time, 'd': d, 'r': optimal_retention}))

    fig = px.line(df2, x="retention", y="expected time", color='d', log_y=True)

    # print(f"\n-----suggested retention: {np.inner(difficulty_distribution_padding, optimal_retention_list):.2f}-----")
    suggested_retention_markdown = f"""# Suggested Retention: `{np.inner(difficulty_distribution_padding, optimal_retention_list):.2f}`"""
    return fig, suggested_retention_markdown
