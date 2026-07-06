"""Run the four founding scenarios and print full inference traces."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from apm import Engine
from scenarios.fixtures import NOW, s1_iaeste, s2_movie, s3_call, s4_travel

CASES = [
    ("S1  'Don't forget Friday the 26th!' (fallback ladder)", s1_iaeste()),
    ("S1b same query, agenda deleted (the gate must refuse)", s1_iaeste(with_agenda=False)),
    ("S2  movie recall from 3 partial cues (intersection)", s2_movie()),
    ("S2b movie recall from 1 cue (tie -> recency default)", s2_movie(all_cues=False)),
    ("S3  'call from Person X' (pre-emptive, predictive)", s3_call()),
    ("S4  'he goes to ^£*#>' (corrupted -> tagged default)", s4_travel()),
]

for title, (g, q) in CASES:
    print("=" * 78)
    print(title)
    print("-" * 78)
    v = Engine(g).answer(q, now=NOW)
    print(v.render())
    print()
