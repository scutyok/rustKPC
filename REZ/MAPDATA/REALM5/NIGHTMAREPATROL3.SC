Name "nightmarepatrol3"

AbortOnDamage			FALSE		
AbortOnEnemySighted		FALSE		
CanBeMoved			FALSE
SightRange				1600.0		
TimeOut					999.0
WalkSpeed				140
YawRate					180
CheckForEnemyInterval	0.5			
TargetPlayer						

StartAnim	"AmbA"
Wait  10
MoveToPos 1664 448 -2048
Wait  0.2
StartAnim	"Fly"
WalkSpeed				300
MoveToPos 1664 256 -1600


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
	SightRange	2400.0

TurretAttack:

	OnEnemySighted_Goto		DISABLE
	OnDamage_Goto			DISABLE
	OnEnemyOutOfRange_Goto	        Loop1
	
	YawRate		360.0

	TurnToTarget	0.6	
	RangedAttackTarget
        MeleeAttackTarget


        Goto	               TurretAttack


