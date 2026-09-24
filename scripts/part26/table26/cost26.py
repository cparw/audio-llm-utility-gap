"""PART26 pod cost. Read-only: billing rows come from GET https://rest.runpod.io/v1/billing/pods (saved by curl to
billing_p26_window.json), pod ids and rates from the PART26 create responses (pods_found.json), delete times from the
watchdog .gone markers and sidecars. CPU pods have no rows in the billing API, so they get rate x (create to delete)."""
import json, sys
from datetime import datetime
b = json.load(open(sys.argv[1])); pods = json.load(open("pods_found.json"))
DEL = {  # delete time UTC on 24 Sep, from wd/*.gone, try2/wd/*.gone and sidecars
 "mspkz2hwf76o9f": "01:08:07", "hkgitzvp7db9z9": "01:43:49", "wikwa0qc0uw1b9": "03:35:36", "mmqahrpwpu0m00": "03:36:03",
 "yz8nyovce4nx08": "01:28:22", "m42ljldnq92qmw": "01:26:59", "vz0e628ewxabop": "01:18:49", "pcg8t6ahr6bo4q": "01:28:26",
 "roj4j50j5vbk1a": "01:37:33", "ykow87gaqraiqs": "01:18:37",
 "j0g9l1t6s9r3vm": "00:55:47", "md42nsos0qf0ei": "00:59:30", "l7pkzosi9j1glg": "00:55:14", "k93fsuqg1hk1y7": "00:57:07",
 "dow5f8gprrxhi2": "00:51:14", "fx4m1lrp4exk6p": "00:53:04", "vyn4hnge5s8jn0": "00:55:08", "x8hcwgu4l1zzie": "00:54:44",
 "fecp2sxone9uvn": "00:54:35", "psewpwt8nok4c0": "00:55:01", "6cy4hpql72vvio": "01:08:54", "orph1m3rsbw8u2": "00:58:02",
 "bkrhdn09yvpf3h": "00:58:27", "marzjz3lnmpdhs": "00:55:06", "n37xm21htbr44z": "01:42:52", "k9dkn8qzhw51a6": "02:52:04",
 "p6afmxz5ilpzr1": "01:22:07", "1b33rynkogtm3y": "01:39:40", "6zo60zbviepd7a": "02:48:16", "boyfheojko3eqb": "01:24:59",
 "xm65txjjvqzq5q": None,  # found gone at 02:40; delete time not recorded
 "o0ezblvilljrg0": "02:54:40", "bg8a6dnjgyjyyh": "02:51:12", "rvk1syvmw4l0x2": "01:27:07", "5dt4v6o0nffj1o": "03:42:36",
}
def t(s): return datetime.strptime("2026-09-24 " + s, "%Y-%m-%d %H:%M:%S")
billed = {}
for r in b:
    if r["podId"] in pods:
        x = billed.setdefault(r["podId"], [0.0, 0])
        x[0] += r["amount"]; x[1] += r["timeBilledMs"]
rows = []; tot = {"gpu_billed": 0, "gpu_est": 0, "cpu_est": 0, "cpu_est_max_unknown": 0}
for pid, v in pods.items():
    created = v["created"][11:19]
    rate = float(v["cost"])
    gpu = rate > 2
    d = DEL[pid]
    mins = (t(d) - t(created)).total_seconds() / 60 if d else None
    est = rate * mins / 60 if mins is not None else None
    bl = billed.get(pid)
    rows.append((v["cell"], v["name"], pid, "GPU" if gpu else "CPU", rate, created, d or "unknown (gone by 02:40)", mins, est, bl[0] if bl else None, bl[1] / 1000 if bl else None))
    if gpu:
        tot["gpu_est"] += est; tot["gpu_billed"] += bl[0] if bl else 0
    else:
        if est is None:
            tot["cpu_est_max_unknown"] += rate * (t("02:40:00") - t(created)).total_seconds() / 3600
        else:
            tot["cpu_est"] += est
rows.sort(key=lambda r: (r[0], r[5]))
for r in rows:
    print("\t".join(str(round(x, 3)) if isinstance(x, float) else str(x) for x in r))
print(json.dumps({k: round(v, 2) for k, v in tot.items()}))
json.dump({"rows": rows, "totals": tot}, open("cost26_out.json", "w"), indent=1)
