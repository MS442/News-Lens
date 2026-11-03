"""The learned component: evidence + claim -> three NLI relation scores.

Training is in training/train_evidence.py. Production uses the exported weights
in Python ONNX Runtime; no hosted language model supplies the verdict.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
LABELS = ("contradiction", "entailment", "neutral")
# Second, lower tier used when no passage clears the strict thresholds.
# Contradiction needs more confidence: NLI models over-predict it for passages
# that are merely about a different event on the same topic.
LIKELY_THRESHOLDS = {"entailment": 0.60, "contradiction": 0.75}
LIKELY_MARGIN = 0.25
REVERSE_CONTRADICTION_MIN = 0.5


class EvidenceClassifier:
    def __init__(self, model_dir=None):
        folder = Path(model_dir or ROOT / "models/evidence")
        self.config = json.loads((folder / "runtime_config.json").read_text(encoding="utf-8"))
        if self.config.get("evaluation_pending"):
            raise RuntimeError("The evidence model is still being evaluated. Please try again later.")
        if self.config.get("label_order") != list(LABELS):
            raise RuntimeError("The model label order is not compatible with this application.")
        for filename, expected in (("model.onnx", self.config["sha256"]),
                                   ("tokenizer.json", self.config["tokenizer_sha256"])):
            with (folder / filename).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != expected:
                raise RuntimeError("The evidence model files failed their integrity check.")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.max_tokens = int(self.config["max_tokens"])
        self.tokenizer.enable_truncation(max_length=self.max_tokens, strategy="longest_first")
        self.tokenizer.no_padding()  # Each pair is scored individually, just like evaluation.
        self.lock = threading.Lock()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(folder / "model.onnx"), options, providers=["CPUExecutionProvider"])
        check = json.loads((folder / "runtime_self_test.json").read_text(encoding="utf-8"))
        if check["model_sha256"] != self.config["sha256"]:
            raise RuntimeError("The runtime verification belongs to a different model.")
        for case in check["cases"]:
            ids = np.asarray([case["input_ids"]], dtype=np.int64)
            mask = np.asarray([case.get("attention_mask", [1] * len(case["input_ids"]))], dtype=np.int64)
            actual = self.session.run(["logits"], {"input_ids": ids, "attention_mask": mask})[0][0]
            # Dynamic int8 kernels differ slightly between CPU generations (AVX2 vs
            # AVX-512/VNNI). Allow that drift but still catch a wrong/corrupt model.
            tolerance = max(float(check["absolute_tolerance"]), 0.25)
            if not np.allclose(actual, case["logits"], atol=tolerance, rtol=0) or \
                    int(np.argmax(actual)) != int(np.argmax(case["logits"])):
                raise RuntimeError("The evidence model failed its CPU compatibility check. Predictions are temporarily unavailable.")

    def compare(self, evidence: str, claim: str):
        result = self._compare(evidence, claim)
        if result["relation"] != "contradiction":
            return result
        # A real contradiction is symmetric ("in Paris" vs "in Berlin" conflict
        # either way round). NLI models also emit "contradiction" when a passage
        # merely describes a different detail of the same story; those cases are
        # not symmetric. Confirm with the reversed pair before counting it.
        reverse = self._compare(claim, evidence)
        result["reverse_contradiction"] = reverse["scores"]["contradiction"]
        if reverse["scores"]["contradiction"] < REVERSE_CONTRADICTION_MIN:
            result.update(relation="review", decision="review",
                          decision_reason="Tentative contradiction not confirmed in reverse: the passage likely describes a different detail.")
        return result

    def _compare(self, evidence: str, claim: str):
        with self.lock:
            encoded = self.tokenizer.encode(evidence, claim)
        feed = {"input_ids": np.asarray([encoded.ids], dtype=np.int64),
                "attention_mask": np.asarray([encoded.attention_mask], dtype=np.int64)}
        logits = self.session.run(["logits"], feed)[0][0].astype(np.float64)
        logits /= float(self.config["temperature"])
        values = np.exp(logits - logits.max())
        values /= values.sum()
        scores = dict(zip(LABELS, map(float, values)))
        predicted = LABELS[int(values.argmax())]
        ordered = np.sort(values)
        margin = float(ordered[-1] - ordered[-2])
        confidence = float(values.max())
        strong = self.config["thresholds"]
        likely = {**LIKELY_THRESHOLDS, **self.config.get("likely_thresholds", {})}
        likely_margin = float(self.config.get("likely_minimum_margin", LIKELY_MARGIN))
        if predicted != "neutral" and confidence >= strong[predicted] and margin >= self.config["minimum_margin"]:
            decision, reason = "decisive", "High confidence: the validation-selected threshold and margin are met."
        elif predicted != "neutral" and confidence >= likely[predicted] and margin >= likely_margin:
            # Real news passages are longer and noisier than the FEVER-style
            # validation pairs, so true matches rarely clear the 95%-precision
            # thresholds. A clear (but lower) lead still counts as moderate evidence.
            decision, reason = "likely", "Moderate confidence: the model clearly leans this way, below the strict threshold."
        else:
            decision = "review"
            reason = ("The model predicts neutral: this passage does not establish the statement." if predicted == "neutral" else
                      "The model has a tentative prediction, but it is too weak to count as evidence.")
        return {"scores": scores, "relation": predicted if decision != "review" else "review",
                "decision": decision, "decision_reason": reason,
                "raw_relation": predicted, "confidence": confidence, "margin": margin,
                "truncated": bool(encoded.overflowing), "token_count": len(encoded.ids)}
