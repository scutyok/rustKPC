Name "cm3patrol2"

AbortOnDamage FALSE
AbortOnEnemySighted FALSE
SightRange 1200
TimeOut 999.0
FaceMoveDir TRUE
YawRate 180
FieldOfView 70
WalkSpeed 70
CheckForEnemyInterval	0.5			
TargetPlayer			

StartAnim "Run"
MoveToPos -576 560 900
TurnToPos -576 560 1456
MoveToPos -576 560 1456
TurnToPos -768 560 1456
Goto		Loop1
			
Loop1:
	LoopAnim	TRUE
	StartAnim	"AmbA"
	OnEnemyOutOfRange_Goto	DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			FarAttack
	Wait		999.0
	Goto		Loop1

FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange
	      
BoostSightRange:
	SightRange	2048.0
Goto		Loop1
TurretAttack:
	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        Loop1
	YawRate		360.0
	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget
	If_HealthLessThanHalf_Goto	RunAway
	Goto				TurretAttack

RunAway:
StartAnim "Run"
TurnToPos -576 560 1200
MoveToPos -576 560 1200
TurnToPos -576 560 1456
LoopAnim	TRUE
StartAnim	"AmbA"
	
Goto				TurretAttack2
TurretAttack2:
	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        DISABLE

	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget
Goto				TurretAttack2

