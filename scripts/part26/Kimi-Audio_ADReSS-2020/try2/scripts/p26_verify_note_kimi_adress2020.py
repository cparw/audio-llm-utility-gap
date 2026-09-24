"""PART26 Kimi-Audio ADReSS-2020: add the verification summary to the sidecar and write a note on the one strict check that
failed. p26_verify_kimi_adress2020.py passed 16 of 17 checks. The failed one, mac_refit_at_saved_layers_same_auc, refits
every outer fold on the Mac at the pod's chosen layers and asks for the same AUC within 1e-9. The Mac runs sklearn 1.7.2,
scipy 1.15.3, numpy 2.2.6 with Apple BLAS, not the pinned pod stack (sklearn 1.9.1, numpy 2.1.2, scipy 1.18.1), so the
lbfgs fits differ slightly (the script docstring says "not expected to be bit equal"). The pod-side podverify, on the pinned
stack, refit at the saved layers with max prediction difference 0 and re-derived every inner layer choice (0 mismatches).
The Mac inner layer choice also equals the pod in all 30 outer folds."""
import json, numpy as np, datetime
C = "scores/part26/Kimi-Audio_ADReSS-2020"
V = json.load(open(f"{C}/verify/p26_verify_kimi_adress2020.json"))
S = json.load(open(f"{C}/verify/p26_states_check_kimi_adress2020.json"))
fails = [k for k, v in V["checks"].items() if not v["ok"]]
m = V["checks"]["mac_refit_at_saved_layers_same_auc"]["per_tag"]
mac5 = float(np.mean([m[f"seed{i}"]["auc_mac_refit"] for i in range(5)])); pod5 = float(np.mean([m[f"seed{i}"]["auc_pod"] for i in range(5)]))
note = {"date_utc": datetime.datetime.utcnow().isoformat() + "Z", "verify_checks": len(V["checks"]), "failed": fails,
        "failed_reason": "Mac refit on sklearn 1.7.2 / scipy 1.15.3 / numpy 2.2.6 (Apple BLAS), not the pinned pod stack; lbfgs fits differ slightly",
        "mac_refit_auc_minus_pod": {t: v["auc_mac_refit"] - v["auc_pod"] for t, v in m.items()},
        "mac_refit_max_abs_pred_diff": {t: v["max_abs_pred_diff"] for t, v in m.items()},
        "mac_refit_mean5": mac5, "pod_mean5": pod5, "mean5_diff": mac5 - pod5, "same_at_4dp": round(mac5, 4) == round(pod5, 4),
        "pod_podverify": V["checks"]["podverify_pass"], "mac_inner_layer_choice": V["checks"]["mac_inner_layer_choice_equals_pod_up_to_exact_ties"]["ok"],
        "states_check_all_pass": S["states_ok"]}
json.dump(note, open(f"{C}/verify/p26_verify_kimi_adress2020_note.json", "w"), indent=1)
P = f"{C}/per_clip/p26_kimi_adress2020_perclip.sidecar.json"; sc = json.load(open(P))
sc["verification"] = {"states_check": f"{C}/verify/p26_states_check_kimi_adress2020.json (11 of 11 pass)",
                      "independent_mac_check": f"{C}/verify/p26_verify_kimi_adress2020.json ({len(V['checks']) - len(fails)} of {len(V['checks'])} pass)",
                      "note": f"{C}/verify/p26_verify_kimi_adress2020_note.json",
                      "only_failure": "mac_refit_at_saved_layers_same_auc: Mac refit on sklearn 1.7.2 differs from the pinned pod by at most "
                                      f"{max(abs(v) for v in note['mac_refit_auc_minus_pod'].values()):.1e} AUC per repeat, {abs(mac5 - pod5):.1e} on mean5 (same at 4 dp); "
                                      "the pod-side podverify on the pinned stack is exact (max prediction difference 0, 0 layer mismatches)"}
json.dump(sc, open(P, "w"), indent=1)
print(json.dumps(note, indent=1))
