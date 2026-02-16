# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

"""Export tab: save locally, upload to Hugging Face, or chat with the model."""

from __future__ import annotations

import gradio as gr


def create_export_page(pipeline_state: gr.State) -> dict:
    """Create the Export tab with save, upload, and chat functionality."""

    gr.Markdown("## Export Model")
    gr.Markdown(
        "Save the abliterated model locally, upload it to Hugging Face Hub, or chat with it "
        "to verify it works as expected. **Important:** Select a trial in the Results tab first."
    )

    # Save locally
    with gr.Accordion("Save to Local Folder", open=True):
        save_path = gr.Textbox(
            label="Save Path",
            placeholder="/path/to/output/folder",
        )
        save_btn = gr.Button("Save Model", variant="primary")
        save_status = gr.Markdown("")

    # Upload to HF
    with gr.Accordion("Upload to Hugging Face", open=False):
        with gr.Row():
            hf_token = gr.Textbox(
                label="HF Access Token",
                type="password",
                placeholder="hf_...",
            )
            hf_repo = gr.Textbox(
                label="Repository Name",
                placeholder="username/model-name-clawly",
            )
        hf_private = gr.Checkbox(label="Private Repository", value=False)
        upload_btn = gr.Button("Upload Model", variant="primary")
        upload_status = gr.Markdown("")

    # Chat interface
    gr.Markdown("### Chat with Model")
    gr.Markdown("Test the abliterated model interactively.")

    chatbot = gr.Chatbot(label="Chat", height=400, type="messages")
    with gr.Row():
        chat_input = gr.Textbox(
            label="Message",
            placeholder="Type a message...",
            scale=4,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)
    clear_btn = gr.Button("Clear Chat")

    def save_model(path, state):
        if state is None:
            return "Pipeline not configured."
        if not path:
            return "Please enter a save path."
        try:
            state.save_model(path, strategy="merge")
            return f"Model saved to `{path}`."
        except Exception as e:
            return f"Error: {e}"

    def upload_model(token, repo, private, state):
        if state is None:
            return "Pipeline not configured."
        if not token or not repo:
            return "Please provide both token and repository name."
        try:
            state.upload_model(repo, token, private=private, strategy="merge")
            return f"Model uploaded to `{repo}`."
        except Exception as e:
            return f"Error: {e}"

    def chat_respond(message, history, state):
        if state is None:
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "Pipeline not configured. Please configure and run optimization first."})
            return "", history

        history = history or []
        history.append({"role": "user", "content": message})

        # Build messages for the model
        messages = [
            {"role": "system", "content": state.settings.system_prompt},
        ]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Stream response
        response_text = ""
        try:
            for token in state.stream_chat(messages):
                response_text += token

            history.append({"role": "assistant", "content": response_text})
        except Exception as e:
            history.append({"role": "assistant", "content": f"Error: {e}"})

        return "", history

    def clear_chat():
        return []

    save_btn.click(
        save_model,
        inputs=[save_path, pipeline_state],
        outputs=[save_status],
    )

    upload_btn.click(
        upload_model,
        inputs=[hf_token, hf_repo, hf_private, pipeline_state],
        outputs=[upload_status],
    )

    send_btn.click(
        chat_respond,
        inputs=[chat_input, chatbot, pipeline_state],
        outputs=[chat_input, chatbot],
    )

    chat_input.submit(
        chat_respond,
        inputs=[chat_input, chatbot, pipeline_state],
        outputs=[chat_input, chatbot],
    )

    clear_btn.click(clear_chat, outputs=[chatbot])

    return {
        "chatbot": chatbot,
        "save_btn": save_btn,
        "upload_btn": upload_btn,
    }
