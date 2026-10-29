# R16b label guide: operator requests for the OpenSwarm ground station (v2)

Each item is one line of JSON:

```json
{"id": "b001", "text": "...", "expect": "ok", "params": {"behavior": "...", ...}, "tags": ["..."]}
```

`expect` is `ok` (the station should send this command), `clarify` (it should ask a question) or `refuse` (it should decline). `params` appears only for `ok`. Write `params` **with only the values the operator stated**. The station fills in the defaults below, and scoring compares the filled forms.

## The one-command rule

A request maps to exactly **one** of eight swarm behaviours. They apply to the whole swarm, never to one drone.

| behavior | What it does | Params (unit), default when unstated, allowed range |
|---|---|---|
| `hover` | hold position | `alt_m` (m). Default: hold the current altitude. Range 0.5–30. |
| `formation` | fly in a shape | `shape`: `line`, `v`, `grid` or `circle` (default `v`). `spacing_m`, the distance between neighbours (default 1.5, range 0.8–10). `alt_m` (0.5–30). |
| `boids` | flock freely like birds | `cohesion` (default 1.0), `separation` (default 1.5), `alignment` (default 1.0); each a weight from 0 to 5. |
| `search_grid` | sweep an area in lanes, looking for people or things | `area` (a place, **mandatory**). `alt_m` (default 5). |
| `perimeter` | circle a place to watch or guard it | `area` (a place, **mandatory**). `radius_m` (default 10, range 2–100). `alt_m` (default 5). |
| `person_follow` | follow the person the scout camera detects | `distance_m` (default 3, range 1–10). `alt_m` (default 2). `min_conf`, the detection confidence needed, as a fraction (default 0.7, range 0.3–0.99; "80%" means 0.8). |
| `light_show` | synchronised LED show | `pattern`: `rainbow`, `pulse`, `chase`, `sparkle` or `wave` (default `rainbow`). `bpm`, an integer (default 120, range 40–200). |
| `rth` | return to home | none |

## Places

The site map. Use the canonical name in `params.area`; the operator may say any alias.

| Canonical name | Aliases |
|---|---|
| home | base, launch point, launch pad, here |
| north field | the field to the north |
| car park | parking lot, parking |
| river bank | riverbank, the river |
| barn | the old barn |
| east woods | the woods, forest |
| orchard | apple orchard |
| school | schoolyard |
| highway | the road, motorway |

`school` and `highway` are no-fly areas. A later safety check refuses them. Translation must still produce them as `ok` with that area: the translator's job is to say what was asked.

## Units and numbers

- Convert feet to metres (1 ft = 0.3048 m) and round to 2 decimals. Example: 20 ft → 6.1.
- Percentages become fractions for `min_conf` only.
- Words count as numbers ("two metres" → 2), including fractions and decimals ("two and a half metres" → 2.5, "one point five" → 1.5, "half a metre" → 0.5).
- "at N m" with no word naming another setting means the altitude ("hover at 12 m" → `alt_m` 12). A word that names the setting wins: "at 4 m distance" is the follow distance, "20 m out" or "radius 20 m" is the perimeter radius, "3 m apart" is the spacing.
- A value outside its range (e.g. spacing 0.5 m, altitude 50 m, bpm 300) → `refuse`. The station must not fly a value the operator did not ask for.

## When to clarify

- `search_grid` or `perimeter` with no place, or with a place that is not on the map (e.g. "the lake", "over there", "the hill").
- The request clearly wants the swarm to act, but it is not clear which behaviour, e.g. "do the thing from yesterday" or "get into position".

## When to refuse

- The request is not about flying the swarm (weather, jokes, battery report requests, settings).
- Actions none of the behaviours does: land, take off, film or record, take photos, spray, deliver, drop something, fly to a coordinate, a move by one named drone, or speed changes.
- Two behaviours at once or in sequence ("hover and then do a light show"), or contradictions ("line up in a circle").
- A value outside its range (see above).
- A parameter the behaviour does not have, e.g. an altitude for `boids`, or spacing for `perimeter`.

## Coverage

Aim for this mix:
- about 60% `ok`, covering every behaviour, with params stated and omitted, aliases, feet, and number words;
- about 20% `clarify`;
- about 20% `refuse`.

Vary the style: terse radio-call style ("all units, RTH"), casual ("guys come on back"), long with context ("we've got a missing hiker, last seen near the old barn, can you circle it at 20 m out"), typos, and polite or urgent tone. Each text must be unique. Do not copy wording from this guide.

Put useful tags on each item: the behaviour, `alias`, `feet`, `default`, `range`, `unknown-place`, `no-place`, `out-of-domain`, `two-behaviours`, `single-drone`, `wrong-param`, `no-fly`, `typo`.
