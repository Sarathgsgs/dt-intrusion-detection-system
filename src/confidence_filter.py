"""
Milestone 6: Operational Confidence Filter Module
Suppresses low-fidelity false positives and ambiguous alerts by validating
model confidence against domain-informed attack feature signatures.
"""

import os
import sys
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Domain Attack Feature Signatures for Industrial IoT
ATTACK_SIGNATURES: Dict[str, List[str]] = {
    "DDoS_UDP": ["udp.stream", "udp.time_delta", "udp.port", "tcp.len"],
    "DDoS_ICMP": ["icmp.checksum", "icmp.seq_le", "tcp.flags"],
    "DDoS_HTTP": ["http.content_length", "http.response", "tcp.dstport", "tcp.flags.ack"],
    "DDoS_TCP": ["tcp.flags", "tcp.connection.syn", "tcp.connection.synack", "tcp.len"],
    "Port_Scanning": ["tcp.dstport", "tcp.srcport", "tcp.flags", "tcp.connection.syn"],
    "SQL_injection": ["http.content_length", "http.response", "tcp.len", "tcp.payload"],
    "XSS": ["http.content_length", "http.response", "tcp.len"],
    "Ransomware": ["tcp.len", "tcp.seq", "tcp.ack", "mbtcp.len"],
    "Vulnerability_scanner": ["tcp.dstport", "tcp.flags", "http.content_length", "udp.port"],
    "Password": ["http.response", "tcp.len", "tcp.flags", "tcp.dstport"],
    "Backdoor": ["tcp.dstport", "tcp.srcport", "tcp.flags.ack", "tcp.len"],
    "Uploading": ["http.content_length", "tcp.len", "tcp.flags"],
    "Fingerprinting": ["tcp.flags", "tcp.connection.syn", "tcp.options"],
    "MITM": ["arp.opcode", "arp.hw.size", "tcp.flags"]
}

class OperationalConfidenceFilter:
    def __init__(self, min_confidence: float = 0.65, min_signature_overlap: int = 1):
        self.min_confidence = min_confidence
        self.min_signature_overlap = min_signature_overlap
        self.signatures = ATTACK_SIGNATURES
        self.stats = {
            "total_inspected": 0,
            "passed_alerts": 0,
            "suppressed_alerts": 0,
            "normal_traffic": 0
        }
        
    def evaluate(self, prediction_result: Dict[str, Any], shap_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates whether an alert should be passed to SOC / operator dashboard or suppressed.
        """
        self.stats["total_inspected"] += 1
        predicted_class = prediction_result.get("predicted_class", "Normal")
        confidence = prediction_result.get("confidence", 0.0)
        
        # If predicted Normal, no alert needed
        if predicted_class == "Normal":
            self.stats["normal_traffic"] += 1
            return {
                "decision": "NORMAL",
                "should_alert": False,
                "reason": "Traffic classified as normal baseline behavior",
                "confidence": confidence,
                "signature_overlap": 0,
                "matching_features": []
            }
            
        # Check domain signature match
        expected_features = self.signatures.get(predicted_class, [])
        top_shap_features = [f["feature"] for f in shap_explanation.get("top_features", []) if f.get("shap_value", 0) > 0]
        
        matching_features = [f for f in top_shap_features if any(exp in f for exp in expected_features)]
        overlap_count = len(matching_features)
        
        # Decision Logic:
        # 1. High confidence (> min_conf) AND signature match -> PASS
        # 2. Low confidence OR zero signature match -> SUPPRESS
        if confidence >= self.min_confidence and overlap_count >= self.min_signature_overlap:
            decision = "PASS"
            should_alert = True
            reason = f"High-fidelity alert: Confidence ({confidence*100:.1f}%) meets threshold and features {matching_features} match known domain mechanics."
            self.stats["passed_alerts"] += 1
        else:
            decision = "SUPPRESS"
            should_alert = False
            reasons = []
            if confidence < self.min_confidence:
                reasons.append(f"Low model confidence ({confidence*100:.1f}% < {self.min_confidence*100:.1f}%)")
            if overlap_count < self.min_signature_overlap:
                reasons.append("SHAP attributions diverge from expected domain attack signature")
            reason = f"Alert suppressed: {'; '.join(reasons)}."
            self.stats["suppressed_alerts"] += 1
            
        return {
            "decision": decision,
            "should_alert": should_alert,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "signature_overlap": overlap_count,
            "matching_features": matching_features,
            "reason": reason
        }

def test_confidence_filter():
    print("Testing OperationalConfidenceFilter...")
    filter_engine = OperationalConfidenceFilter(min_confidence=0.70, min_signature_overlap=1)
    
    # Case 1: High Confidence Clear DDoS Alert
    case_clear_ddos = {
        "pred": {"predicted_class": "DDoS_UDP", "confidence": 0.94},
        "shap": {"top_features": [
            {"feature": "udp.stream", "shap_value": 0.85},
            {"feature": "udp.port", "shap_value": 0.42},
            {"feature": "tcp.flags", "shap_value": -0.10}
        ]}
    }
    res1 = filter_engine.evaluate(case_clear_ddos["pred"], case_clear_ddos["shap"])
    print("\nCase 1 (Clear DDoS Alert):")
    print(f"Decision: {res1['decision']} | Alert: {res1['should_alert']} | Reason: {res1['reason']}")
    assert res1["decision"] == "PASS", "Expected clear attack to PASS"
    
    # Case 2: Ambiguous / Borderline Alert (Low Confidence)
    case_ambiguous = {
        "pred": {"predicted_class": "SQL_injection", "confidence": 0.52},
        "shap": {"top_features": [
            {"feature": "arp.opcode", "shap_value": 0.22},
            {"feature": "tcp.ack", "shap_value": 0.15}
        ]}
    }
    res2 = filter_engine.evaluate(case_ambiguous["pred"], case_ambiguous["shap"])
    print("\nCase 2 (Ambiguous Alert):")
    print(f"Decision: {res2['decision']} | Alert: {res2['should_alert']} | Reason: {res2['reason']}")
    assert res2["decision"] == "SUPPRESS", "Expected ambiguous alert to be SUPPRESSED"
    
    print("\n[SUCCESS] Confidence Filter successfully validated on all test cases!")

if __name__ == "__main__":
    test_confidence_filter()
