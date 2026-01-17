Name "Headless Hop"

AbortOnDamage TRUE			//	abort the script if we're damaged
AbortOnEnemySighted FALSE	//	Don't abort if we sight an enemy
SightRange 512.0			//	can only see 512 units
TimeOut 999.0

//	wait for a bit
Wait 2.0
Walk

//	walk out in front of the player
StartAnim "Walk"
FaceMoveDir	TRUE
MoveToPos -1666 72 1189

//	notice the player and freak out
StartAnim "AmbB_Old"
Wait 0.1
YawRate 360
FacePos -1600 72 832
Wait 0.9

//	run away
Run
FaceMoveDir	TRUE
StartAnim "Walk"
MoveToPos -1536 72 1848
MoveToPos -1282 72 1848

StartAnim "AmbA"
TurnToPos -1536 72 1848
End
