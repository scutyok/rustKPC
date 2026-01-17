Name "turret02b"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
SightRange			2400.0		//  can see 
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0
CheckForEnemyInterval		0.5		//  check for enemies every 1/2 second
TargetPlayer					//  target the player
CanBeMoved			FALSE		//  don't allow outside forces to push the monster


///////////////////////////////////////////////////////////////////////////////
//	Move ballbuster out to turret position
///////////////////////////////////////////////////////////////////////////////
MoveToSpot:
	LoopAnim	True
	StartAnim 	"run"
	run
	FaceMoveDir 	TRUE
	MoveTo 		t02pos01
	TurnTo		t02aim01
	StartAnim 	"AmbB"
	StartAnim	"AmbA"
	Goto		TurretAttack
	

///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	OnEnemyOutOfRange_Goto		DISABLE
	OnEnemySighted_Goto		TurretAttack
	OnDamage_Goto			TurretAttack

	LoopAnim	TRUE
	StartAnim	"AmbA"

	//	Just wait a while... Monster will still check for targets, so
	//	OnXXX_Goto's will break us out of this when an enemy sights us 
	//	or shoots us
	Wait		999.0
	Goto		Loop1

///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
TurretAttack:
	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		Loop1
	
	TurnToTarget	0.57	
	RangedAttackTarget

	LoopAnim	TRUE
	StartAnim	"AmbA"
	wait 1.0

	//	if we're close to dying, increase our rate of fire
	If_HealthLessThanHalf_Goto	Crazy
	Goto				TurretAttack

///////////////////////////////////////////////////////////////////////////////
//	When ballbuster is almost dead he goes Crazy.
///////////////////////////////////////////////////////////////////////////////
Crazy:
	LoopAnim	FALSE

	OnDamage_Goto			DISABLE
	OnEnemySighted_Goto		DISABLE
	OnEnemyOutOfRange_Goto		Loop1
	
	TurnToTarget			0.57	
	RangedAttackTarget
		
	Goto				Crazy

