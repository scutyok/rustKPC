Name "cratebuster2"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
SightRange			2400.0		//  can see 
TimeOut				999.0
WalkSpeed			70
YawRate				120
CheckForEnemyInterval		0.5		//  check for enemies every 1/2 second
TargetPlayer					//  target the player


///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	LoopAnim	TRUE
	StartAnim	"AmbA"

	OnEnemyOutOfRange_Goto		DISABLE
//	OnEnemySighted_Goto		TurretAttack
//	OnDamage_Goto			TurretAttack
	Goto				TurretAttack
	
	Wait	100000.0
	Goto	Loop1


///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
TurretAttack:
	LoopAnim	FALSE

	OnDamage_Goto			MoveToSpot
	OnEnemyOutOfRange_Goto		Loop1
	
	YawRate		360.0
	TurnToTarget	0.57	
	RangedAttackTarget
	LoopAnim	TRUE
	StartAnim	"AmbA"
	wait .5

	//  make sure ballbuster can still see the player.
	If_TargetNOTVisible_Goto	Loop1
	If_HealthLessThanQuarter_Goto	Crazy
	Goto				TurretAttack


///////////////////////////////////////////////////////////////////////////////
//	If ballbuster gets hit he moves back to turret spot.
///////////////////////////////////////////////////////////////////////////////
MoveToSpot:
	LoopAnim	True
	StartAnim 	"Walk"
	FaceMoveDir 	TRUE
	MoveTo 		turretspot2
	StartAnim 	"AmbA"
	wait .5
	Goto Loop1


///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Crazy:
	LoopAnim	FALSE

	OnDamage_Goto			MoveToSpot
	OnEnemyOutOfRange_Goto		Loop1
	
	YawRate		360.0

	TurnToTarget	0.57	
	RangedAttackTarget
		
	//  make sure ballbuster can still see the player.
	If_TargetNOTVisible_Goto	Loop1
	Goto				Crazy

