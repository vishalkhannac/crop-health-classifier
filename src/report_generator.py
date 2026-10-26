# src/report_generator.py — Exportable PDF Health & Diagnostic Report Generator
import io
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import textwrap


def generate_pdf_report(pil_img: Image.Image, analysis: dict, gradcam_img: Image.Image = None) -> bytes:
    """
    Generates a clean, professional single-page PDF diagnostic report using Matplotlib's PdfPages
    (zero extra external dependencies required; works 100% offline).
    Returns raw PDF bytes ready for st.download_button().
    """
    buf = io.BytesIO()
    items = analysis.get("items", [])
    summary = analysis.get("summary", {})
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69), dpi=150)  # A4 Portrait
        fig.patch.set_facecolor("#ffffff")

        # Header banner
        ax_header = fig.add_axes([0.05, 0.89, 0.90, 0.08])
        ax_header.set_facecolor("#1b5e20")
        ax_header.set_xticks([])
        ax_header.set_yticks([])
        for spine in ax_header.spines.values():
            spine.set_visible(False)
        ax_header.text(
            0.03, 0.62, "PLANT & PRODUCE HEALTH DIAGNOSTIC REPORT",
            color="white", fontsize=14, fontweight="bold", transform=ax_header.transAxes
        )
        ax_header.text(
            0.03, 0.22, f"Generated: {timestamp_str}  |  AI Model: MobileNetV2 (70-Class Fine-Tuned)",
            color="#c8e6c9", fontsize=9, transform=ax_header.transAxes
        )

        # Annotated Input Image + Optional Grad-CAM Image
        annotated = analysis.get("annotated_image", pil_img)
        if gradcam_img is not None:
            ax_img1 = fig.add_axes([0.06, 0.61, 0.42, 0.25])
            ax_img1.imshow(annotated)
            ax_img1.set_title("Analyzed Specimen / Localization", fontsize=10, fontweight="bold", color="#1b5e20", pad=6)
            ax_img1.axis("off")

            ax_img2 = fig.add_axes([0.52, 0.61, 0.42, 0.25])
            ax_img2.imshow(gradcam_img)
            ax_img2.set_title("Grad-CAM AI Attention Heatmap", fontsize=10, fontweight="bold", color="#0d47a1", pad=6)
            ax_img2.axis("off")
        else:
            ax_img1 = fig.add_axes([0.22, 0.61, 0.56, 0.25])
            ax_img1.imshow(annotated)
            ax_img1.set_title("Analyzed Specimen", fontsize=10, fontweight="bold", color="#1b5e20", pad=6)
            ax_img1.axis("off")

        # Body Diagnostic Content
        ax_body = fig.add_axes([0.06, 0.08, 0.88, 0.50])
        ax_body.axis("off")

        y_cursor = 0.96

        def draw_line(label, value, color="#212121", bold_val=True, step=0.055):
            nonlocal y_cursor
            ax_body.text(0.01, y_cursor, label, fontsize=10, fontweight="bold", color="#424242", transform=ax_body.transAxes, va="top")
            wrapped = "\n".join(textwrap.wrap(str(value), width=70))
            n_lines = wrapped.count("\n") + 1
            ax_body.text(
                0.28, y_cursor, wrapped,
                fontsize=10, fontweight=("bold" if bold_val else "normal"),
                color=color, transform=ax_body.transAxes, va="top"
            )
            y_cursor -= max(step, n_lines * 0.036 + 0.015)

        if summary.get("is_multi") and len(items) > 1:
            draw_line("Inspection Mode:", f"Multi-Object Batch Analysis ({len(items)} items detected)", "#0d47a1")
            draw_line("Batch Breakdown:", f"{summary.get('fresh_count', 0)} Fresh / Safe  |  {summary.get('rotten_count', 0)} Rotten / Diseased", "#1b5e20" if summary.get("batch_safe") else "#c62828")
            draw_line("Batch Verdict:", summary.get("batch_verdict", ""), "#1b5e20" if summary.get("batch_safe") else "#b71c1c")

            y_cursor -= 0.02
            ax_body.text(0.01, y_cursor, "INDIVIDUAL OBJECT BREAKDOWN:", fontsize=10, fontweight="bold", color="#1b5e20", transform=ax_body.transAxes, va="top")
            y_cursor -= 0.045

            for item in items[:6]:
                v = item["verdict_info"]
                safe_str = "SAFE" if v["safe"] else "NOT SAFE"
                line_txt = f"#{item['index']}: {v['status']} ({item['confidence']:.1f}%) — [{safe_str}] {v['verdict']}"
                ax_body.text(0.03, y_cursor, line_txt, fontsize=9.5, fontweight="bold", color=("#2e7d32" if v["safe"] else "#c62828"), transform=ax_body.transAxes, va="top")
                y_cursor -= 0.035
                if v.get("treatment"):
                    adv_wrapped = "\n".join(textwrap.wrap(f"Action: {v['treatment']}", width=85))
                    ax_body.text(0.05, y_cursor, adv_wrapped, fontsize=8.5, color="#555555", transform=ax_body.transAxes, va="top")
                    y_cursor -= (adv_wrapped.count("\n") + 1) * 0.03 + 0.015
        elif items:
            item = items[0]
            v = item["verdict_info"]
            safe_color = "#2e7d32" if v["safe"] else "#c62828"

            draw_line("Identified Class:", v["status"], "#0d47a1")
            draw_line("AI Confidence:", f"{item['confidence']:.1f}%", "#1b5e20" if item["confidence"] >= 70 else "#e65100")
            draw_line("Disease / Condition:", v["disease"], "#880e4f")
            draw_line("Consumption Verdict:", v["verdict"], safe_color)
            draw_line("Diagnostic Note:", v["note"], "#333333", bold_val=False)
            draw_line("Recommended Action:", v.get("treatment", "N/A"), "#1b5e20", bold_val=False)
            draw_line("Storage & Prevention:", v.get("prevention", "N/A"), "#0d47a1", bold_val=False)

            if item.get("top3"):
                top_str = "  |  ".join([f"{t['status']} ({t['confidence']:.1f}%)" for t in item["top3"]])
                draw_line("Top-3 Candidates:", top_str, "#555555", bold_val=False)

        # Footer Disclaimer
        ax_footer = fig.add_axes([0.05, 0.02, 0.90, 0.045])
        ax_footer.set_facecolor("#f5f5f5")
        ax_footer.set_xticks([])
        ax_footer.set_yticks([])
        for spine in ax_footer.spines.values():
            spine.set_edgecolor("#e0e0e0")
        ax_footer.text(
            0.5, 0.5,
            "DISCLAIMER: Visual estimate from exterior appearance only — not a laboratory microbiological food-safety test.",
            fontsize=8, color="#616161", ha="center", va="center", style="italic", transform=ax_footer.transAxes
        )

        pdf.savefig(fig)
        plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
