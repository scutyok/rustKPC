Name "nightmareturret3"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
CanBeMoved			FALSE
SightRange				1600.0		
TimeOut					999.0
WalkSpeed				70
YawRate					180
CheckForEnemyInterval	0.5			
TargetPlayer						



Loop1:
	LoopAnim	TRUE
	StartAnim	"AmbA"
	OnEnemyOutOfRange_Goto	DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			FarAttack
	Wait		1

	Goto		Loop1

FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange

	Goto		Loop1
       
BoostSightRange:
	SightRange	2400.0

TurretAttack:
	LoopAnim	FALSE

	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	Loop1
	
	YawRate		360.0

	TurnToTarget	0.57	
	RangedAttackTarget
        MeleeAttackTarget

        LoopAnim	TRUE
        StartAnim	"AmbA"
        Wait            1.0

	If_TargetNOTVisible_Goto		Loop1

	Goto				TurretAttack

