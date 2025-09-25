### ✅ **Expected Trends and Validation**

#### 1. **Cut Layer 3 (Late Cut) Performs Best**

* **Clean accuracy reaches 45–48%** — expected since most model depth is on the client.
* **ASR increases with attacker percentage**, especially for well-structured patterns (e.g., `Static case`, `Pattern Invariant`).
* **Backdoor accuracy drops** with increasing ASR, showing the model is more confidently misclassifying toward the target label.

> ✅ *This is typical and aligns with prior SplitFL research findings.*

---

#### 2. **Cut Layer 1 (Early Cut) Struggles**

* **Clean and backdoor accuracy \~10%** → suggests underfitting or insufficient feature propagation.
* **ASR often stays at 0%** even at 50% attackers → likely the backdoor pattern is erased in early layers.

> ⚠️ *Expected due to low expressive power in the client model.*

---

#### 3. **Cut Layer 2 Is Inconsistent**

* Clean accuracy sometimes improves (13–18%), and ASR spikes in a few configs.
* For `Static case` cut at 2: ASR = 94.77% (with just 1 attacker!) — this might be an outlier or an overfitting case.
* At 50%, ASR drops again — possibly due to instability or collapsed model weights.

> ⚠️ *Needs deeper inspection. You could try adding weight norm clipping or tweaking learning rate to improve stability.*

---

### ⚠️ **Potential Anomalies to Investigate**

| Observation                                                    | Possible Cause                                                     | Suggested Fix                                                                          |
| -------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| High ASR at 0% attackers (`Location Invariant`, `cut_layer=1`) | Possibly unintentional trigger-like artefacts or leakage           | Double-check trigger injection logic to ensure it’s only applied when `malicious=True` |
| ASR = 0% at high attacker %, some configs                      | Ineffective pattern (especially for randomised sizes or placement) | Try visualising backdoor samples via `save_backdoor_images()`                          |
| Clean accuracy below 15% in `cut_layer=2`                      | Server not learning enough                                         | Consider using deeper server-side layers or training it more aggressively              |

---

### 📊 Overall Takeaways

* **Cut layer position has a major impact** on both clean performance and ASR.
* **Static and Pattern Invariant triggers** work best — especially at cut layer 3.
* **Randomised triggers are less effective**, as expected, but still lead to moderate ASR at deeper cuts.
* **More attackers generally lead to higher ASR**, but sometimes collapse occurs at low client count (NUM\_CLIENTS=5).
