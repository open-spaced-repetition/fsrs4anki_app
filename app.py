import gradio as gr
import pytz
import os
from datetime import datetime
from markdown import instructions_markdown, faq_markdown
from fsrs4anki_optimizer import Optimizer
from pathlib import Path
from utilities import cleanup

def get_w_markdown(w):
    return f"""
    # Updated Parameters
    Copy and paste these as shown in step 5 of the instructions:

    `{w}`
    
    Check out the Analysis tab for more detailed information."""


def anki_optimizer(file: gr.File, timezone, next_day_starts_at, revlog_start_date, requestRetention,
                   progress=gr.Progress(track_tqdm=True)):
    now = datetime.now()
    files = ['prediction.tsv', 'revlog.csv', 'revlog_history.tsv', 'stability_for_analysis.tsv',
             'expected_time.csv', 'evaluation.tsv']
    prefix = now.strftime(f'%Y_%m_%d_%H_%M_%S')
    suffix = file.name.split('/')[-1].replace(".", "_").replace("@", "_")
    proj_dir = Path(f'projects/{prefix}/{suffix}')
    proj_dir.mkdir(parents=True, exist_ok=True)
    print(proj_dir)
    os.chdir(proj_dir)
    proj_dir = Path('.')
    optimizer = Optimizer()
    optimizer.anki_extract(file.name)
    analysis_markdown = optimizer.create_time_series(timezone, revlog_start_date, next_day_starts_at).replace("\n", "\n\n")
    optimizer.define_model()
    optimizer.train()
    w_markdown = get_w_markdown(optimizer.w)
    optimizer.predict_memory_states()
    difficulty_distribution = optimizer.difficulty_distribution.to_string().replace("\n", "\n\n")
    plot_output = optimizer.find_optimal_retention()[0]
    suggested_retention_markdown = f"""# Suggested Retention: `{optimizer.optimal_retention:.2f}`"""
    rating_markdown = optimizer.preview(requestRetention).replace("\n", "\n\n")
    loss_before, loss_after = optimizer.evaluate()
    loss_markdown = f"""
**Loss before training**: {loss_before}

**Loss after training**: {loss_after}
    """
    # optimizer.calibration_graph()
    # optimizer.compare_with_sm2()
    markdown_out = f"""{suggested_retention_markdown}

# Loss Information
{loss_markdown}

# Difficulty Distribution
{difficulty_distribution}

# Ratings
{rating_markdown}
"""
    files_out = [file for file in files if (proj_dir / file).exists()]
    cleanup(proj_dir, files)
    return w_markdown, markdown_out, plot_output, files_out


description = """
# FSRS4Anki Optimizer App - v3.24.1
Based on the [tutorial](https://medium.com/@JarrettYe/how-to-use-the-next-generation-spaced-repetition-algorithm-fsrs-on-anki-5a591ca562e2) 
of [Jarrett Ye](https://github.com/L-M-Sherlock). This application can give you personalized anki parameters without having to code.

Read the `Instructions` if its your first time using the app.
"""

with gr.Blocks() as demo:
    with gr.Tab("FSRS4Anki Optimizer"):
        with gr.Box():
            gr.Markdown(description)
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    file = gr.File(label='Review Logs (Step 1)')
                with gr.Column():
                    next_day_starts_at = gr.Number(value=4,
                                                   label="Next Day Starts at (Step 2)",
                                                   precision=0)
                    timezone = gr.Dropdown(label="Timezone (Step 3.1)", choices=pytz.all_timezones)
                    with gr.Accordion(label="Advanced Settings (Step 3.2)", open=False):
                        requestRetention = gr.Number(value=.9, label="Desired Retention: Recommended to set between 0.8  0.9")
                        revlog_start_date = gr.Textbox(value="2006-10-05",
                                                       label="Revlog Start Date: Optimize review logs after this date.")
        with gr.Row():
            btn_plot = gr.Button('Optimize your Anki!')
        with gr.Row():
            w_output = gr.Markdown()
    with gr.Tab("Instructions"):
        with gr.Box():
            gr.Markdown(instructions_markdown)
    with gr.Tab("Analysis"):
        with gr.Row():
            markdown_output = gr.Markdown()
            with gr.Column():
                plot_output = gr.Plot()
                files_output = gr.Files(label="Analysis Files")
    with gr.Tab("FAQ"):
        gr.Markdown(faq_markdown)

    btn_plot.click(anki_optimizer,
                   inputs=[file, timezone, next_day_starts_at, revlog_start_date, requestRetention],
                   outputs=[w_output, markdown_output, plot_output, files_output])

demo.queue().launch(show_error=True)
