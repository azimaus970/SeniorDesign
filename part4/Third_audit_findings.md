# Third audit of the Part IV doc (2026-11-05)

Checked every number against its source, every Part III reference against the Part III text, every code claim against the code, the citations against their arXiv abstracts (read only, nothing downloaded), and the page setup against the SDP template.

## Confirmed correct (no change)
- Every Table 5.3 value matches `ai-workload/benchmark/benchmark_results.json`. That covers parameters, top-1, top-5, FPS, p50/p95 and peak RAM. The "up to 21%" FPS difference matches the first-pass file. "2.8×" latency and "1.9×" memory recompute correctly.
- The application results match `ai-workload/results/`: 412/500, 375/500, the 7.4-point drop, 3.2 × 10⁻⁶, 160,502 inferences and 89.2 FPS.
- The Part III references are correct: 60–120 s windows, 16–64 units, the three-period hold, 35/40/45 °C, set_profile/get_qos, the E2 retention rule, the Pi-side steady-state run for E0 and the 8 GB board (Part III team instructions). The 4 GB and 100 ms limits are your decisions.
- The cited claims match their abstracts: Benoit-Cattin (seconds, ~90%, 27.7%, 22–36 °C, Pi 4B), Marasini & Zhou (Pi 5, frame rate and energy per frame), Cerqueira, Howard, Baller and Shanks.
- The page setup matches the template's body sections: A4, 1.5 in sides and bottom, 1.2 in top, the same footer. No rough notes or placeholders were found.

## Fixed
| Where | Problem | Fix |
|---|---|---|
| Inference application | Mean latency "11.16 ms" was the plain average of per-second means, not the inference-weighted mean that Table 5.2 defines | Now 11.04 ms, inference-weighted, computed from the same log |
| Inference application | "the four cores at 100%" | "system-wide CPU use averaging 100% of the four cores" (logged mean 99.99%) |
| Pi 5 memory | "247.0 to 247.5 MB, with no growth" contradicted itself | It now says the value started at 247.0 MB, reached 247.5 MB at about 9 min, and did not rise after that |
| **Code: predictor selection** | **Real bug.** Part III 4.3.3.2 requires M3 and M4 to share model type, horizon and **window**. The code picked the window separately for each variant | One shared window per model: the one with the lowest validation RMSE averaged over both variants. Each variant then takes its best setting at that window. New end-to-end assertion added |
| **Code: C3 test** | C3 claimed an improvement from the bootstrap interval alone. This is the same weakness that S01 fixed for D1: with 3 test runs it could claim, even though no exact test can go below p = 0.25 | It now also needs the exact sign-flip Wilcoxon test (Pratt) at p < 0.05. Two new tests added; **26 tests pass** (`ml-pipeline/test_record.txt`) |
| Table 5.1 C1 | Prerequisite omitted B1, which Figure 5.1 shows | Added |
| Table 5.5 AM0 | Depended on the "preliminary benchmark", which was withdrawn in the first review | Now the benchmark on the benchmark machine (Table 5.3) |
| Table 5.5 AM1 | set_profile()/get_qos() are provided as write_control()/read_qos() | Stated |
| Table 5.5 AM2 | Said "feature builder shared with FeatureBuffer" and "metric scripts". FeatureBuffer is not written, and no D1 metric code exists | Now "feature builder following the FeatureBuffer rules" and "C1–C3 metric scripts" |
| Table 5.5 AM7 | No milestone wrote the Table 5.2 analysis script | Added: written and frozen before the first comparison run |
| Values fixed by the team / AM5 | Listed "the window" as a team value, but it is selected on the validation runs | The team fixes the primary horizon, and the shared window is selected |

## Decided by Azim (2026-11-05)
- Shared-window rule approved: one window per model for both variants, the one with the lowest validation RMSE averaged over the two variants.
