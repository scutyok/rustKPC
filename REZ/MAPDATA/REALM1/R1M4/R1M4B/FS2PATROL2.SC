Name "fs2patrol2"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange				1024.0		//	can see 1024 units
TimeOut					999.0
WalkSpeed				70
YawRate					180
CheckForEnemyInterval	0.5			//	check for enemies every 1/2 second
TargetPlayer						//	target the player
CanBeMoved			FALSE
///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
	LoopAnim	TRUE
	StartAnim	"AmbA"
	OnEnemyOutOfRange_Goto	DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			FarAttack
	Wait		999.0

	Goto		Loop1

//	if we get hit before the enemy is in range
FarAttack:
	If_DamageFromPlayer_Goto	BoostSightRange
	
	Goto		Loop1

BoostSightRange:
	SightRange	2048.0

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
	If_HealthLessThanQuarter_Goto	Crazy

	Goto				TurretAttack

Crazy:
	LoopAnim	FALSE

	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		Loop1
	
	TurnToTarget			0.57	
	RangedAttackTarget
		
	Goto				Crazy

