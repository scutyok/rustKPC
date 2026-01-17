Name "hedgehog"

AbortOnDamage TRUE
AbortOnEnemySighted FALSE
Loop TRUE
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180

Loop1:
Walk
StartAnim "Walk"
MoveToPos 350 72 100
TurnToPos -480 72 -140
StartAnim "AmbB"
Wait 1.0
StartAnim "Walk"
MoveToPos -480 72 -140
TurnToPos -30 72 230
StartAnim "AmbB"
Wait 1.0
StartAnim "Walk"
MoveToPos -30 72 230
TurnToPos -700 72 370
StartAnim "AmbB"
Wait 1.0
StartAnim "Walk"
MoveToPos -700 72 370
TurnToPos -480 72 -380
StartAnim "AmbB"
Wait 1.0
Run
StartAnim "Run"
MoveToPos -480 72 -140
TurnToPos 350 72 100
StartAnim "AmbB"
Wait 1.0
GoTo Loop1