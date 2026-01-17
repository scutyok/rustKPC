Name "BallBuster Patrol"

AbortOnDamage TRUE			//	abort the script if we're damaged
AbortOnEnemySighted FALSE	//	Don't abort if we sight an enemy
SightRange 512.0			//	can only see 512 units
TimeOut 999.0
WalkSpeed 70
YawRate 180

Loop1:
StartAnim "Walk"
TurnToPos -1608 72 1896
MoveToPos -1608 72 1896
TurnToPos -1600 72 832
StartAnim "AmbA"
Wait 2.0

Run
StartAnim "Run"
MoveToPos -1600 72 1000

StartAnim "AmbA"


Wait 2.0

StartAnim "Deflect"
Wait 1.1

Walk
StartAnim "Walk"
TurnToPos -1600 72 1896
MoveToPos -1600 72 1896
StartAnim "AmbA"
Wait 2.0

StartAnim "Walk"
TurnToPos -1280 72 1896
FaceMoveDir TRUE
MoveToPos  -1280 72 1896
StartAnim "AmbA"
Wait 2.0
Goto Loop1
