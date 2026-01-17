Name "ArachniTurret"

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

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
	LoopAnim	TRUE
	StartAnim	"AmbA"
	OnEnemyOutOfRange_Goto	DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			FarAttack
	Wait		100000.0

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

	//	this is the amount of time from the start of the Arachiniclown's attack
	//	to the point where he starts firing.  Get this value from ModelEdit
	TurnToTarget	0.57	
	RangedAttackTarget

	//	make sure we can still see the target
	If_TargetNOTVisible_Goto		Loop1
	If_HealthLessThanQuarter_Goto	Finish

//	StartAnim	"AmbA"
//	Wait		3.0

	Goto	TurretAttack

Finish:
	End

