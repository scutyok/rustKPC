Name "cmturret2"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
CanBeMoved			FALSE
SightRange				1200.0	
FieldofView	                        180
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


