Name "sb2patrol2"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
CanBeMoved			FALSE
SightRange	1024
TimeOut 999.0
FaceMoveDir TRUE
YawRate 90
FieldOfView 70
CheckForEnemyInterval	0.5			
TargetPlayer						


Loop1:

OnEnemyOutOfRange_Goto	DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			FarAttack

Walk
StartAnim "Walk"
MoveToPos 1230 160 2500
TurnToPos 1200 160 3104
StartAnim "AmbA"
Wait 0.6
StartAnim "Walk"
MoveToPos 1200 160 3104
TurnToPos 1230 160 2500
StartAnim "AmbA"
Wait 0.6

GoTo Loop1

FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange
	
	Goto		Loop1

BoostSightRange:
	SightRange	2048.0

TurretAttack:

	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	Loop1
	
	YawRate		360.0

	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget

Goto				TurretAttack
