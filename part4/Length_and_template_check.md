# Full-length and template check (2026-11-05)

## 1. Was anything cut for length?
I compared every sentence of all 15 earlier builds of the doc with the current one. Each removed sentence falls into one of these groups:
- **Removed under your no-made-up-numbers rule (kept out).** Planned run counts (about 30 runs, 12 cycles, 4 lab days), the 0.9–1.2 GB memory estimate, the 4 GB vs 8 GB price comparison, the XGBoost/GRU/LSTM grid values, E2 chunk length and noise settings, CMP 491 week numbers, and the synthetic simulator.
- **Replaced by a correction from the reviews.** The old benchmark numbers, the "highest accuracy" claim, the EfficientNet "export fault", the nearest-second log join, "PCM hardware effect", "time-weighted accuracy", the 50 ms per-call limit and the stall wording.
- **Reworded, with the content still in the doc.**
- **Cut without a reason. These are now restored.**
  - Accuracy is top-1 over **all 1,000 ImageNet classes**, not only the 10 subset classes. This is true of the benchmark code.
  - Why CIFAR-10 was not used: its 32 px images and its classes do not match ImageNet.
  - Why ONNX Runtime was chosen: the workload and the predictors share one runtime, and the Pi needs no training library. PyPI's 64-bit Arm Linux wheel for onnxruntime 1.31.1 / Python 3.13 was **checked by downloading it today**.
  - The images are processed in the same fixed order every run, so the inputs are identical and no disk reads enter the timing. This was checked in the code.
  - set_profile() changes the load without restarting the program.
  - Marasini and Zhou's comparison used sustained 30-minute workloads. This is checked against the arXiv abstract.

## 2. Format compared against the template's XML
| Item | Template | Doc before | Now |
|---|---|---|---|
| Page size, margins, header/footer distance | A4; top 1728, sides 2160, bottom 2160 twips (body sections); footer2 centred page number | Same | Same (no change) |
| Body text | Normal: Times (majorBidi), 12 pt, justified, 1.25 line spacing, 6 pt after | Same | Same |
| Heading 3 | Heading3 style with numbering switched off and the number typed, as in "4.3.1 System Architecture" | Same | Same |
| Heading 4 | Heading4 style (bold, colour 983222) | Same | Same |
| Caption label | **"Table 5.1." in bold**, plain text after it, ending with a full stop. Figures use **"Figure 5.1"** in bold with no full stop after the number | Whole caption plain, no bold label, no final full stop | **Fixed**: bold label and full stop, figure label as in the template |
| Table style | TableGrid1 | Table Grid | **Fixed** |
| Table cell text | Normal style with direct single spacing and 0 after; Times New Roman 10 pt, character spacing −0.1 pt, bold header, centred number columns | Body Text style, 10 pt, all left | **Fixed** to match exactly, including centred ID/number columns |
| Table width | tblW in dxa | auto, centred | **Fixed**: width in dxa, not centred (same position, as the table spans the text width) |
| References | "References" in Abstract style, entries in Bibliography style | Normal, left-aligned, 0.4 in hanging | **Fixed**: Bibliography style (justified, like the template) with Word's 0.5 in APA hanging indent |
| Empty paragraph after each table | Yes | Yes | Same |
| Picture paragraph | Centred, keep with next | Same | Same |

**Left as is, on purpose.** Caption numbers are typed rather than Word fields. The template's fields (STYLEREF 1 + SEQ) take the chapter number from a Heading 1, and this standalone doc has none. As fields they would show the wrong numbers until the doc is pasted into the full report.

The doc is now 20 pages instead of 22. Nothing was removed; tables take less room because their cells now use the template's single line spacing.
