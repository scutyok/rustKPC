Name "BallBuster Patrol"

AbortOnDamage TRUE			//	abort the script if we're damaged
AbortOnEnemySighted FALSE		//	Don't abort if we sight an enemy
SightRange 512.0			//	can only see 512 units
TimeOut 999.0
WalkSpeed 70
YawRate 180

Loop1:
StartAnim "Walk"
//TurnTo CNodeLink6
TurnToAngle 180
MoveTo CNodeLink6
//TurnTo CTriggerBrush_Generic0
TurnToAngle 270
StartAnim "AmbA"
Wait 2.0

Run
StartAnim "Run"
MoveTo CTriggerBrush_Generic0

StartAnim "AmbA"


Wait 2.0

StartAnim "Deflect"
Wait 1.1

Walk
StartAnim "Walk"
//TurnTo CNodeLink6
TurnToAngle 90
MoveTo CNodeLink6
StartAnim "AmbA"
Wait 2.0

StartAnim "Walk"
//TurnTo CNodeLink7
TurnToAngle 0
FaceMoveDir TRUE
MoveTo CNodeLink7
StartAnim "AmbA"
Wait 2.0
Goto Loop1
