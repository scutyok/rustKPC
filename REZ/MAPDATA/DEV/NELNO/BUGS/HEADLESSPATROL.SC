Name "ScripWar_Headless"

AbortOnDamage			TRUE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange				512.0		//	can only see 512 units
TimeOut					999.0
WalkSpeed				160
YawRate					180
FaceMoveDir				TRUE		//	face where we're moving top
CheckForEnemyInterval	999999.0
AlertNear				FALSE
AlertGroup				TRUE
GroupAlertRange			2048.0

Loop1:

Walk
StartAnim "Walk"
//TurnToPos	head_path3
MoveTo		head_path3

StartAnim "Walk"
//TurnToPos	head_path2
MoveTo		head_path2

StartAnim "Walk"
//TurnToPos	head_path1
MoveTo		head_path1

StartAnim "Walk"
//TurnToPos	head_path4
MoveTo		head_path4

Goto	Loop1



