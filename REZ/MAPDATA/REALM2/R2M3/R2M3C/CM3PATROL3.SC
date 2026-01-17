Name "cm3patrol3"

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

StartAnim "Walk"
MoveToPos 608 256 -368
TurnToPos 608 256 0

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

	Goto		Loop1      
BoostSightRange:
	SightRange	2048.0

TurretAttack:
	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        Loop1
	
	YawRate		360.0

	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget


        Goto	               TurretAttack

