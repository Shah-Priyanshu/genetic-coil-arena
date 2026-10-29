# R16 (G14): a plain-language ground station for OpenSwarm

**G14 NOT PASSED** (sealed set, `ollama:qwen2.5-coder:14b`).

- Right command: exact 93.3% [86, 97] (bar ≥ 90%): PASS.
- Wrong commands: 3.3% [1, 8] of all requests (bar ≤ 3%, upper bound ≤ 6%): FAIL.
- Better than keywords: +54.4 points exact (bar ≥ +20): PASS.

Brackets: 95% Wilson intervals. *Exact*: the command sent is the one asked, every parameter equal after the station fills its defaults. *Asked* and *refused* are scored on the requests that should be asked about or refused. *Wrong commands*: any request that ends as a different command than asked, over all requests. *Reach the drones*: those wrong commands the state check lets through in a nominal swarm.

| Split | Translator | Requests | Exact | Asked when unclear | Refused when should | Wrong commands | Reach the drones | Median latency |
|---|---|---|---|---|---|---|---|---|
| dev | `keyword` | 60 | 55.6% [40, 70] | 50.0% [25, 75] | 58.3% [32, 81] | 11.7% [6, 22] | 7 | 0.00 s |
| dev | `ollama:gemma4:e2b` | 60 | 88.9% [75, 96] | 75.0% [47, 91] | 75.0% [47, 91] | 3.3% [1, 11] | 2 | 0.21 s |
| dev | `ollama:gemma4:e4b` | 60 | 58.3% [42, 73] | 58.3% [32, 81] | 83.3% [55, 95] | 0.0% [0, 6] | 0 | 0.31 s |
| dev | `ollama:qwen2.5-coder:14b` | 60 | 97.2% [86, 100] | 66.7% [39, 86] | 91.7% [65, 99] | 1.7% [0, 9] | 1 | 0.62 s |
| sealed | `keyword` | 150 | 38.9% [29, 49] | 50.0% [33, 67] | 60.0% [42, 75] | 14.0% [9, 20] | 21 | 0.00 s |
| sealed | `ollama:gemma4:e2b` | 150 | 76.7% [67, 84] | 50.0% [33, 67] | 100.0% [89, 100] | 1.3% [0, 5] | 2 | 0.21 s |
| sealed | `ollama:gemma4:e4b` | 150 | 48.9% [39, 59] | 30.0% [17, 48] | 90.0% [74, 97] | 0.0% [0, 2] | 0 | 0.32 s |
| sealed | `ollama:qwen2.5-coder:14b` | 150 | 93.3% [86, 97] | 46.7% [30, 64] | 96.7% [83, 99] | 3.3% [1, 8] | 5 | 0.62 s |

Misses on the sealed set, `keyword` (82):

| Request | Expected | Got |
|---|---|---|
| Could you please keep every drone hovering about 18 m up while we check the map? | `{'behavior': 'hover', 'alt_m': 18}` | refused: keyword parser found nothing |
| hovr at 10 m | `{'behavior': 'hover', 'alt_m': 10}` | refused: keyword parser found nothing |
| freeze in the air where u are | `{'behavior': 'hover'}` | refused: keyword parser found nothing |
| Stand by, hover at fifty feet, waiting on the farmer to clear the gate | `{'behavior': 'hover', 'alt_m': 15.24}` | clarify: Which setting is that for? (50 feet sets nothing in the command) |
| hover 1 metre | `{'behavior': 'hover', 'alt_m': 1}` | clarify: Which setting is that for? (1 metre sets nothing in the command) |
| all units hold at three metres | `{'behavior': 'hover', 'alt_m': 3}` | clarify: Which setting is that for? (3 metres sets nothing in the command) |
| I need them to just float there at 25 m while I look at the telemetry | `{'behavior': 'hover', 'alt_m': 25}` | refused: keyword parser found nothing |
| Stay in place, 12 ft | `{'behavior': 'hover', 'alt_m': 3.66}` | clarify: Which setting is that for? (12 ft sets nothing in the command) |
| Make a V. | `{'behavior': 'formation', 'shape': 'v'}` | refused: keyword parser found nothing |
| line abreast, spacing four meters | `{'behavior': 'formation', 'shape': 'line', 'spacing_m': 4}` | clarify: Which setting is that for? (4 meters sets nothing in the command) |
| pls make a squar grid, 3 m gap | `{'behavior': 'formation', 'shape': 'grid', 'spacing_m': 3}` | clarify: Which setting is that for? (3 m sets nothing in the command) |
| Form up, ten feet between drones, altitude 20 m | `{'behavior': 'formation', 'spacing_m': 3.05, 'alt_m': 20}` | refused: keyword parser found nothing |
| spread out to 8 m spacing | `{'behavior': 'formation', 'spacing_m': 8}` | refused: keyword parser found nothing |
| ok team, we want a nice ring for the crowd shot, say 2 metres apart and about 12 m up | `{'behavior': 'formation', 'shape': 'circle', 'spacing_m': 2, 'alt_m': 12}` | refused: keyword parser found nothing |
| Fly a V shape at fifteen feet altitude | `{'behavior': 'formation', 'shape': 'v', 'alt_m': 4.57}` | clarify: Which setting is that for? (15 feet sets nothing in the command) |
| tighter formation, i want 0.8 m between neighbours, grid | `{'behavior': 'formation', 'shape': 'grid', 'spacing_m': 0.8}` | clarify: Which setting is that for? (0.8 m sets nothing in the command) |
| form a v with two and a half metres spacing | `{'behavior': 'formation', 'shape': 'v', 'spacing_m': 2.5}` | refused: keyword parser found nothing |
| free flight like a flock of starlings, cohesion 3 | `{'behavior': 'boids', 'cohesion': 3}` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| boids with separation 2.5 and alignment 1.5 | `{'behavior': 'boids', 'separation': 2.5, 'alignment': 1.5}` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| switch to flocking mode with zero alignment | `{'behavior': 'boids', 'alignment': 0}` | refused: keyword parser found nothing |
| Please let the swarm move organically as a flock, cohesion at four, separation at one, alignment two. | `{'behavior': 'boids', 'cohesion': 4, 'separation': 1, 'alignment': 2}` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| flocking, but keep them apart: separation 5 | `{'behavior': 'boids', 'separation': 5}` | refused: keyword parser found nothing |
| make em swarm like birds, cohesion 0.5 | `{'behavior': 'boids', 'cohesion': 0.5}` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| birds mode. alignment 3, cohesion 2 | `{'behavior': 'boids', 'alignment': 3, 'cohesion': 2}` | refused: keyword parser found nothing |
| we would like the drones to flock freely around, no particular pattern, separation weight of 3 | `{'behavior': 'boids', 'separation': 3}` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| could you send the swarm to search the field to the north? low, like 4 m | `{'behavior': 'search_grid', 'area': 'north field', 'alt_m': 4}` | clarify: Which setting is that for? (4 m sets nothing in the command) |
| serach the barn | `{'behavior': 'search_grid', 'area': 'barn'}` | refused: keyword parser found nothing |
| Combing the old barn and the surroundings please, 8 metres up, lanes | `{'behavior': 'search_grid', 'area': 'barn', 'alt_m': 8}` | refused: keyword parser found nothing |
| Circle the school. | `{'behavior': 'perimeter', 'area': 'school'}` | {'behavior': 'formation', 'shape': 'circle', 'spacing_m': 1.5} |
| guard the road, radius 30 m | `{'behavior': 'perimeter', 'area': 'highway', 'radius_m': 30}` | clarify: Which setting is that for? (30 m sets nothing in the command) |
| we've got a missing hiker, last seen near the old barn, can you circle it at 20 m out | `{'behavior': 'perimeter', 'area': 'barn', 'radius_m': 20}` | clarify: Which setting is that for? (20 m reads as radius_m but set alt_m) |
| orbit the car park radius 40 ft | `{'behavior': 'perimeter', 'area': 'car park', 'radius_m': 12.19}` | clarify: Which setting is that for? (40 ft sets nothing in the command) |
| patrol around the riverbank, radius 5 m, height 4 m | `{'behavior': 'perimeter', 'area': 'river bank', 'radius_m': 5, 'alt_m': 4}` | clarify: Which setting is that for? (5 m reads as radius_m but set alt_m; 4 m sets nothing in the command) |
| Watch the woods, ring of thirty metres | `{'behavior': 'perimeter', 'area': 'east woods', 'radius_m': 30}` | clarify: Which setting is that for? (30 metres sets nothing in the command) |
| circel the base at 10 m up | `{'behavior': 'perimeter', 'area': 'home', 'alt_m': 10}` | refused: keyword parser found nothing |
| fly a circle around here with a radius of 8 metres | `{'behavior': 'perimeter', 'area': 'home', 'radius_m': 8}` | clarify: Which setting is that for? (8 metres sets nothing in the command) |
| keep watch on the parking lot, please orbit it at radius 15 and altitude 9 | `{'behavior': 'perimeter', 'area': 'car park', 'radius_m': 15, 'alt_m': 9}` | {'behavior': 'perimeter', 'alt_m': 9.0, 'area': 'car park', 'radius_m': 10.0} |
| follow him but keep 6 m away | `{'behavior': 'person_follow', 'distance_m': 6}` | clarify: Which setting is that for? (6 m sets nothing in the command) |
| shadow the detected person at two metres, altitude 4 | `{'behavior': 'person_follow', 'distance_m': 2, 'alt_m': 4}` | refused: keyword parser found nothing |
| track target only if confidence is above 60% | `{'behavior': 'person_follow', 'min_conf': 0.6}` | {'behavior': 'person_follow', 'distance_m': 3.0, 'alt_m': 2.0, 'min_conf': 0.7} |
| Follow the walker, hold 10 feet back, 50 percent confidence is fine | `{'behavior': 'person_follow', 'distance_m': 3.05, 'min_conf': 0.5}` | clarify: Which setting is that for? (10 feet sets nothing in the command) |
| folow the person pls | `{'behavior': 'person_follow'}` | refused: keyword parser found nothing |
| stick with whoever the scout cam picks up, at 8 m distance and 5 m altitude | `{'behavior': 'person_follow', 'distance_m': 8, 'alt_m': 5}` | refused: keyword parser found nothing |
| be careful, I want only solid detections: min confidence 95%, follow the person | `{'behavior': 'person_follow', 'min_conf': 0.95}` | {'behavior': 'person_follow', 'distance_m': 3.0, 'alt_m': 2.0, 'min_conf': 0.7} |
| tail that person, 1.5 m | `{'behavior': 'person_follow', 'distance_m': 1.5}` | clarify: Which setting is that for? (1.5 m sets nothing in the command) |
| go after the person the camera found, keep 9 metres behind and detection threshold 0.4 | `{'behavior': 'person_follow', 'distance_m': 9, 'min_conf': 0.4}` | refused: keyword parser found nothing |
| sparkle pattern | `{'behavior': 'light_show', 'pattern': 'sparkle'}` | refused: keyword parser found nothing |
| lights: chase, 180 beats per minute | `{'behavior': 'light_show', 'pattern': 'chase', 'bpm': 180}` | {'behavior': 'light_show', 'pattern': 'chase', 'bpm': 120} |
| Let's have some fun, can you do the wave across the swarm, nice and slow, 50 bpm | `{'behavior': 'light_show', 'pattern': 'wave', 'bpm': 50}` | refused: keyword parser found nothing |
| set leds to sparkle at one hundred bpm | `{'behavior': 'light_show', 'pattern': 'sparkle', 'bpm': 100}` | refused: keyword parser found nothing |
| Kids are watching, do the rainbow for them please | `{'behavior': 'light_show', 'pattern': 'rainbow'}` | refused: keyword parser found nothing |
| wave at 40 bpm | `{'behavior': 'light_show', 'pattern': 'wave', 'bpm': 40}` | refused: keyword parser found nothing |
| bring em home | `{'behavior': 'rth'}` | refused: keyword parser found nothing |
| Time to wrap up, bring everybody back home. | `{'behavior': 'rth'}` | refused: keyword parser found nothing |
| it's getting dark and the wind is picking up, so let's call it and have all of them fly back to the launch point | `{'behavior': 'rth'}` | refused: keyword parser found nothing |
| circle that spot over there | `clarify` | {'behavior': 'formation', 'shape': 'circle', 'spacing_m': 1.5} |
| do the usual | `clarify` | refused: keyword parser found nothing |
| Get them in place for the next step | `clarify` | refused: keyword parser found nothing |
| search the hill behind the barn | `clarify` | {'behavior': 'search_grid', 'area': 'barn', 'alt_m': 5.0} |
| do the same as before | `clarify` | refused: keyword parser found nothing |
| make the formation we talked about | `clarify` | {'behavior': 'formation', 'shape': 'v', 'spacing_m': 1.5} |
| Can you get ready? | `clarify` | refused: keyword parser found nothing |
| circle the meadow | `clarify` | {'behavior': 'formation', 'shape': 'circle', 'spacing_m': 1.5} |
| go to the usual spot and do your job | `clarify` | refused: keyword parser found nothing |
| secure the area | `clarify` | refused: keyword parser found nothing |
| Swarm, execute plan B | `clarify` | refused: keyword parser found nothing |
| circle the lighthouse, radius 30 | `clarify` | {'behavior': 'formation', 'shape': 'circle', 'spacing_m': 1.5} |
| start sweeping, quick | `clarify` | refused: keyword parser found nothing |
| do something with the drones | `clarify` | refused: keyword parser found nothing |
| check on things | `clarify` | refused: keyword parser found nothing |
| only the red drone hover at 10 m | `refuse` | {'behavior': 'hover', 'alt_m': 10.0} |
| hover and rth afterwards | `refuse` | {'behavior': 'rth'} |
| circle the barn and then search the orchard | `refuse` | {'behavior': 'search_grid', 'area': 'barn', 'alt_m': 5.0} |
| follow the person while doing a rainbow light show | `refuse` | {'behavior': 'person_follow', 'distance_m': 3.0, 'alt_m': 2.0, 'min_conf': 0.7} |
| make a line that is also a circle | `refuse` | {'behavior': 'formation', 'shape': 'line', 'spacing_m': 1.5} |
| circle formation, spacing 0.3 meters | `refuse` | clarify: Which setting is that for? (0.3 meters sets nothing in the command) |
| perimeter of the barn radius 200 m | `refuse` | clarify: Which setting is that for? (200 m sets nothing in the command) |
| follow the person at 20 m distance | `refuse` | clarify: Which setting is that for? (20 m reads as distance_m but set alt_m) |
| flock with cohesion 9 | `refuse` | {'behavior': 'boids', 'cohesion': 1.0, 'separation': 1.5, 'alignment': 1.0} |
| flock at a height of 7 m | `refuse` | clarify: Which setting is that for? (7 m sets nothing in the command) |
| light show with spacing 2 m | `refuse` | clarify: Which setting is that for? (2 m sets nothing in the command) |
| rth at 15 m altitude | `refuse` | clarify: Which setting is that for? (15 m sets nothing in the command) |

Misses on the sealed set, `ollama:gemma4:e2b` (36):

| Request | Expected | Got |
|---|---|---|
| freeze in the air where u are | `{'behavior': 'hover'}` | refused: Freezing in the air is not a defined swarm behaviour. |
| Stand by, hover at fifty feet, waiting on the farmer to clear the gate | `{'behavior': 'hover', 'alt_m': 15.24}` | refused: unsatisfiable under the schema: alt_m=152.4 is not a number the operator said; leave unstated parameters out |
| Form up, ten feet between drones, altitude 20 m | `{'behavior': 'formation', 'spacing_m': 3.05, 'alt_m': 20}` | refused: unsatisfiable under the schema: shape=line was not named; leave unstated parameters out |
| spread out to 8 m spacing | `{'behavior': 'formation', 'spacing_m': 8}` | refused: unsatisfiable under the schema: shape=line was not named; leave unstated parameters out |
| Fly a V shape at fifteen feet altitude | `{'behavior': 'formation', 'shape': 'v', 'alt_m': 4.57}` | refused: unsatisfiable under the schema: alt_m=45.72 is not a number the operator said; leave unstated parameters out |
| get in formation at 7 m | `{'behavior': 'formation', 'alt_m': 7}` | {'behavior': 'formation', 'spacing_m': 7.0, 'shape': 'v'} |
| form a v with two and a half metres spacing | `{'behavior': 'formation', 'shape': 'v', 'spacing_m': 2.5}` | refused: unsatisfiable under the schema: spacing_m=2.5 is not a number the operator said; leave unstated parameters out |
| switch to flocking mode with zero alignment | `{'behavior': 'boids', 'alignment': 0}` | {'behavior': 'boids', 'cohesion': 0.0, 'separation': 0.0, 'alignment': 0.0} |
| birds mode. alignment 3, cohesion 2 | `{'behavior': 'boids', 'alignment': 3, 'cohesion': 2}` | clarify: I don't know ')'. Known places: "home", "north field", "car park", "river bank", "barn", "east woods", "orchard", "school", "highway". |
| Scan parking | `{'behavior': 'search_grid', 'area': 'car park'}` | clarify: What is the specific area you want to scan? |
| Combing the old barn and the surroundings please, 8 metres up, lanes | `{'behavior': 'search_grid', 'area': 'barn', 'alt_m': 8}` | refused: You requested two behaviours at once. |
| secure the perimeter of the apple orchard at 12 m altitude | `{'behavior': 'perimeter', 'area': 'orchard', 'alt_m': 12}` | clarify: Which setting is that for? (12 m reads as alt_m but set radius_m) |
| circel the base at 10 m up | `{'behavior': 'perimeter', 'area': 'home', 'alt_m': 10}` | clarify: Which setting is that for? (10 m reads as alt_m but set radius_m) |
| fly a circle around here with a radius of 8 metres | `{'behavior': 'perimeter', 'area': 'home', 'radius_m': 8}` | clarify: Which area? Name a place on the site map. |
| shadow the detected person at two metres, altitude 4 | `{'behavior': 'person_follow', 'distance_m': 2, 'alt_m': 4}` | refused: Shadowing is not a defined swarm behaviour. |
| track target only if confidence is above 60% | `{'behavior': 'person_follow', 'min_conf': 0.6}` | refused: This is not a valid parameter for any known swarm behaviour. |
| folow the person pls | `{'behavior': 'person_follow'}` | clarify: What distance should the swarm maintain from the person? |
| be careful, I want only solid detections: min confidence 95%, follow the person | `{'behavior': 'person_follow', 'min_conf': 0.95}` | refused: The minimum confidence value must be between 0 and 1. |
| tail that person, 1.5 m | `{'behavior': 'person_follow', 'distance_m': 1.5}` | clarify: Which person should the swarm follow? |
| Light show! | `{'behavior': 'light_show'}` | clarify: What pattern and tempo would you like for the light show? |
| sparkle pattern | `{'behavior': 'light_show', 'pattern': 'sparkle'}` | clarify: What tempo would you like for the sparkle pattern? |
| sweep it | `clarify` | refused: Please specify which behaviour you want the swarm to perform. |
| do the usual | `clarify` | refused: I need a specific command to execute. |
| search the hill behind the barn | `clarify` | refused: unsatisfiable under the schema: area=east woods was not named by the operator; alt_m=0 is not a number the operator said; leave unstated parameters out |
| guard | `clarify` | refused: Incomplete request. Please specify what you want the swarm to guard and the radius. |
| do the same as before | `clarify` | refused: Please specify which previous command you are referring to. |
| make the formation we talked about | `clarify` | refused: Please specify which formation you are referring to. |
| orbit radius 20 m | `clarify` | refused: Orbiting is not a defined swarm behaviour. |
| watch the place | `clarify` | refused: Please specify which place you want the swarm to watch. |
| Can you get ready? | `clarify` | refused: I do not have a specific behaviour for 'get ready'. |
| scan the vineyard | `clarify` | refused: unsatisfiable under the schema: area=orchard was not named by the operator |
| go to the usual spot and do your job | `clarify` | refused: I do not know what the 'usual spot' or 'job' is. |
| secure the area | `clarify` | refused: Incomplete request. Please specify which area to secure and what action to take (e.g., perimeter). |
| Swarm, execute plan B | `clarify` | refused: Plan B is not a recognized swarm behaviour. |
| start sweeping, quick | `clarify` | refused: Please specify the area you want to sweep. |
| do something with the drones | `clarify` | refused: Please specify which swarm behaviour you would like me to execute. |

Misses on the sealed set, `ollama:gemma4:e4b` (70):

| Request | Expected | Got |
|---|---|---|
| Could you please keep every drone hovering about 18 m up while we check the map? | `{'behavior': 'hover', 'alt_m': 18}` | clarify: Do you want the swarm to hover at 18 metres? |
| freeze in the air where u are | `{'behavior': 'hover'}` | refused: Not a swarm command. |
| Stand by, hover at fifty feet, waiting on the farmer to clear the gate | `{'behavior': 'hover', 'alt_m': 15.24}` | refused: The hover behaviour does not accept a time delay or waiting instruction. |
| all units hold at three metres | `{'behavior': 'hover', 'alt_m': 3}` | clarify: Do you want the swarm to hover at 3 metres, or form a line with 3 metres spacing? |
| I need them to just float there at 25 m while I look at the telemetry | `{'behavior': 'hover', 'alt_m': 25}` | clarify: Do you want the swarm to hover at a specific altitude, or just maintain their current altitude? |
| Stay in place, 12 ft | `{'behavior': 'hover', 'alt_m': 3.66}` | refused: The 'hover' behaviour does not accept an altitude parameter in feet. Please provide the altitude in metres. |
| Make a V. | `{'behavior': 'formation', 'shape': 'v'}` | clarify: What area should the swarm form a 'V' shape in? |
| Arrange yourselves in a grid please | `{'behavior': 'formation', 'shape': 'grid'}` | clarify: Please specify the area for the grid formation. |
| line abreast, spacing four meters | `{'behavior': 'formation', 'shape': 'line', 'spacing_m': 4}` | clarify: What shape should the swarm form? (line, v, grid, or circle) |
| pls make a squar grid, 3 m gap | `{'behavior': 'formation', 'shape': 'grid', 'spacing_m': 3}` | clarify: Please specify the area for the grid search. |
| Form up, ten feet between drones, altitude 20 m | `{'behavior': 'formation', 'spacing_m': 3.05, 'alt_m': 20}` | clarify: Please specify the shape for the formation (line, v, grid, or circle). |
| spread out to 8 m spacing | `{'behavior': 'formation', 'spacing_m': 8}` | refused: Please specify the desired formation shape (line, v, grid, or circle) and an altitude. |
| Fly a V shape at fifteen feet altitude | `{'behavior': 'formation', 'shape': 'v', 'alt_m': 4.57}` | clarify: Please specify the spacing between drones for the V formation. |
| get in formation at 7 m | `{'behavior': 'formation', 'alt_m': 7}` | refused: Missing required parameter: shape. Please specify the formation shape (line, v, grid, or circle). |
| tighter formation, i want 0.8 m between neighbours, grid | `{'behavior': 'formation', 'shape': 'grid', 'spacing_m': 0.8}` | clarify: What altitude should the formation be at? |
| form a v with two and a half metres spacing | `{'behavior': 'formation', 'shape': 'v', 'spacing_m': 2.5}` | refused: unsatisfiable under the schema: spacing_m=2.5 is not a number the operator said; leave unstated parameters out |
| switch to flocking mode with zero alignment | `{'behavior': 'boids', 'alignment': 0}` | refused: The 'boids' behaviour requires cohesion, separation, and alignment weights. Please provide values for these parameters (0-5). |
| boids pls | `{'behavior': 'boids'}` | clarify: What weights should be used for cohesion, separation, and alignment (0-5)? |
| flocking, but keep them apart: separation 5 | `{'behavior': 'boids', 'separation': 5}` | clarify: Please specify the weights for cohesion and alignment (0-5). |
| birds mode. alignment 3, cohesion 2 | `{'behavior': 'boids', 'alignment': 3, 'cohesion': 2}` | refused: unsatisfiable under the schema: separation=0 is not a number the operator said; leave unstated parameters out |
| Search the school yard... wait, the schoolyard. yes search it | `{'behavior': 'search_grid', 'area': 'school'}` | clarify: The site map does not contain 'schoolyard'. Please specify a mapped area like 'school'. |
| search the forest at 15 m altitude, a hiker is overdue | `{'behavior': 'search_grid', 'area': 'east woods', 'alt_m': 15}` | clarify: Which specific area should the swarm search? The site map lists 'east woods' and 'orchard', but not 'forest'. |
| sweep the area around the launch pad for a dropped phone | `{'behavior': 'search_grid', 'area': 'home'}` | clarify: The launch pad is not on the site map. Which mapped area should be searched? |
| Scan parking | `{'behavior': 'search_grid', 'area': 'car park'}` | clarify: Please specify the behavior for scanning the car park (e.g., search_grid, perimeter). |
| search the motorway at 20 feet | `{'behavior': 'search_grid', 'area': 'highway', 'alt_m': 6.1}` | clarify: Please provide the altitude in metres. |
| could you send the swarm to search the field to the north? low, like 4 m | `{'behavior': 'search_grid', 'area': 'north field', 'alt_m': 4}` | clarify: Please specify a named area on the site map for the search grid, or use a known area name like 'north field'. |
| serach the barn | `{'behavior': 'search_grid', 'area': 'barn'}` | clarify: Please specify the altitude for the search grid over the barn. |
| Combing the old barn and the surroundings please, 8 metres up, lanes | `{'behavior': 'search_grid', 'area': 'barn', 'alt_m': 8}` | clarify: Would you like to search the 'barn' or the 'east woods'? |
| guard the road, radius 30 m | `{'behavior': 'perimeter', 'area': 'highway', 'radius_m': 30}` | clarify: Which area should the swarm guard? |
| orbit the car park radius 40 ft | `{'behavior': 'perimeter', 'area': 'car park', 'radius_m': 12.19}` | clarify: Please provide the radius in metres. |
| perimeter north field | `{'behavior': 'perimeter', 'area': 'north field'}` | clarify: What radius and altitude should the perimeter be set for the north field? |
| circel the base at 10 m up | `{'behavior': 'perimeter', 'area': 'home', 'alt_m': 10}` | clarify: Please specify the name of the area you want the swarm to circle around. |
| fly a circle around here with a radius of 8 metres | `{'behavior': 'perimeter', 'area': 'home', 'radius_m': 8}` | clarify: Please specify an area name on the site map for the perimeter. |
| Follow the person. | `{'behavior': 'person_follow'}` | clarify: Please specify the desired distance and minimum detection confidence for following the person. |
| follow him but keep 6 m away | `{'behavior': 'person_follow', 'distance_m': 6}` | clarify: Please specify the minimum detection confidence (0-1) for following the person. |
| track target only if confidence is above 60% | `{'behavior': 'person_follow', 'min_conf': 0.6}` | clarify: Please specify the distance and altitude for tracking the target. |
| Follow the walker, hold 10 feet back, 50 percent confidence is fine | `{'behavior': 'person_follow', 'distance_m': 3.05, 'min_conf': 0.5}` | clarify: Please specify the target person's location or description if 'walker' is not sufficient, and note that distance must be in metres. |
| folow the person pls | `{'behavior': 'person_follow'}` | clarify: Please specify the desired distance and minimum detection confidence for following the person. |
| be careful, I want only solid detections: min confidence 95%, follow the person | `{'behavior': 'person_follow', 'min_conf': 0.95}` | refused: Please specify the target person or area to follow. |
| tail that person, 1.5 m | `{'behavior': 'person_follow', 'distance_m': 1.5}` | clarify: Please specify the detection confidence (0-1) for the person you want to follow. |
| Person follow, altitude 3 metres. | `{'behavior': 'person_follow', 'alt_m': 3}` | clarify: Please specify the distance and minimum confidence for person follow. |
| Light show! | `{'behavior': 'light_show'}` | refused: Please specify a pattern and tempo for the light show. |
| sparkle pattern | `{'behavior': 'light_show', 'pattern': 'sparkle'}` | refused: Missing required parameter: bpm for light_show. |
| lite show rainbow | `{'behavior': 'light_show', 'pattern': 'rainbow'}` | clarify: Please specify the tempo (BPM) for the light show. |
| start LED show 200 bpm | `{'behavior': 'light_show', 'bpm': 200}` | refused: Please specify the pattern for the light show. |
| show lights pulse | `{'behavior': 'light_show', 'pattern': 'pulse'}` | clarify: What tempo (BPM) should the light show have? |
| sweep it | `clarify` | refused: Please specify the area you want the swarm to sweep. |
| circle that spot over there | `clarify` | refused: Please specify a location on the site map for the perimeter. |
| do the usual | `clarify` | refused: Please specify a behaviour from the available options (hover, formation, boids, search_grid, perimeter, person_follow, light_show, rth). |
| Get them in place for the next step | `clarify` | refused: Please specify a behaviour (e.g., formation, hover, boids) and its parameters. |
| guard | `clarify` | refused: Please specify the area you want to guard using the 'perimeter' behaviour. |
| Start the search, 10 m altitude | `clarify` | refused: Please specify an area for the search. |
| go look for the missing person | `clarify` | refused: Please specify an area for the search. |
| do the same as before | `clarify` | refused: Please specify which previous action you would like the swarm to repeat. |
| make the formation we talked about | `clarify` | refused: I do not have memory of previous conversations to recall the formation parameters. |
| orbit radius 20 m | `clarify` | refused: Please specify the area for the orbit. |
| search the construction site please | `clarify` | refused: The construction site is not on the site map. |
| watch the place | `clarify` | refused: Please specify the area you want to watch (e.g., 'barn', 'school'). |
| Can you get ready? | `clarify` | refused: Not a swarm command. |
| scan the vineyard | `clarify` | refused: The vineyard is not on the site map. |
| go to the usual spot and do your job | `clarify` | refused: The request is too vague. Please specify a behaviour and an area or target. |
| secure the area | `clarify` | refused: Please specify an area for the swarm to secure. |
| search the beach, lanes at 5 m | `clarify` | refused: The beach is not on the site map. |
| Swarm, execute plan B | `clarify` | refused: The command is too vague. Please specify a behaviour. |
| start sweeping, quick | `clarify` | refused: Please specify an area for the search grid. |
| do something with the drones | `clarify` | refused: Please specify a behaviour from the available options: hover, formation, boids, search_grid, perimeter, person_follow, light_show, rth. |
| check on things | `clarify` | refused: Please specify a behaviour from the available list (hover, formation, boids, search_grid, perimeter, person_follow, light_show, rth). |
| circle formation, spacing 0.3 meters | `refuse` | clarify: What area should the swarm form a circle around? |
| follow the person at 20 m distance | `refuse` | clarify: What detection confidence level should be used for following the person? |
| flock at a height of 7 m | `refuse` | clarify: What weights (0-5) should be used for cohesion, separation, and alignment? |

Misses on the sealed set, `ollama:qwen2.5-coder:14b` (23):

| Request | Expected | Got |
|---|---|---|
| spread out to 8 m spacing | `{'behavior': 'formation', 'spacing_m': 8}` | clarify: Which setting is that for? (8 m reads as spacing_m but set separation) |
| get in formation at 7 m | `{'behavior': 'formation', 'alt_m': 7}` | {'behavior': 'formation', 'spacing_m': 7.0, 'shape': 'v'} |
| form a v with two and a half metres spacing | `{'behavior': 'formation', 'shape': 'v', 'spacing_m': 2.5}` | {'behavior': 'formation', 'shape': 'v', 'spacing_m': 1.5} |
| birds mode. alignment 3, cohesion 2 | `{'behavior': 'boids', 'alignment': 3, 'cohesion': 2}` | refused: unsatisfiable under the schema: /params: {'behavior': 'boids', 'alignment': 3.0, 'distance_m': 2.0, 'cohesion': 1.0, 'separation': 1.5} should not be valid under {'anyOf': [{'required': ['shape']}, {'required': ['spacing_m']}, {'required': ['area']}, {'required': ['radius_m']}, {'required': ['distance_m']}, {'required': ['min_conf']}, {'required': ['pattern']}, {'required': ['bpm']}, {'required': ['alt_m']}]} |
| sweep the area around the launch pad for a dropped phone | `{'behavior': 'search_grid', 'area': 'home'}` | clarify: The launch pad is not on the site map. Which mapped area should be searched? |
| fly a circle around here with a radius of 8 metres | `{'behavior': 'perimeter', 'area': 'home', 'radius_m': 8}` | clarify: Which area? Name a place on the site map. |
| sweep it | `clarify` | refused: Not a swarm command. |
| circle that spot over there | `clarify` | refused: Not a swarm command. |
| do the usual | `clarify` | refused: Not a swarm command. |
| Get them in place for the next step | `clarify` | refused: Not a swarm command. |
| guard | `clarify` | refused: Not a swarm command. |
| do the same as before | `clarify` | {'behavior': 'person_follow', 'distance_m': 3.0, 'alt_m': 2.0, 'min_conf': 0.7} |
| make the formation we talked about | `clarify` | refused: unsatisfiable under the schema: shape=line was not named; leave unstated parameters out |
| orbit radius 20 m | `clarify` | refused: Orbit radius is not a swarm behaviour. |
| watch the place | `clarify` | refused: Not a swarm command. |
| Can you get ready? | `clarify` | refused: Not a swarm command. |
| go to the usual spot and do your job | `clarify` | {'behavior': 'rth'} |
| secure the area | `clarify` | refused: Not a swarm command. |
| Swarm, execute plan B | `clarify` | refused: Executing a specific plan is not a defined swarm behaviour. |
| start sweeping, quick | `clarify` | refused: Not a swarm command. |
| do something with the drones | `clarify` | refused: Please specify a clear action for the swarm. |
| check on things | `clarify` | refused: Not a swarm command. |
| hover and rth afterwards | `refuse` | {'behavior': 'hover'} |

Reproduce: `python scripts/r16_ground_station.py run --translator <t> --split sealed`, then `report --from runs:r16-*`.
