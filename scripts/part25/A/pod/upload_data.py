"""stream a flat tar (<ds>/<basename>) of the staged audio plus mf csv and sha256 list to stdout"""
import sys, tarfile, io, os
ST, ds = sys.argv[1:3]
tf = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
lines = [l.rstrip("\n").split("\t") for l in open(f"{ST}/{ds}/files.tsv")]
for p, b, h in lines: tf.add(p, arcname=f"{ds}/{b}")
def addbytes(name, data):
    ti = tarfile.TarInfo(name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
addbytes("files.sha256", "".join(f"{h}  {b}\n" for p, b, h in lines).encode())
addbytes(f"mf_{ds}.csv", open(f"{ST}/{ds}/mf_{ds}.csv", "rb").read())
tf.close()
