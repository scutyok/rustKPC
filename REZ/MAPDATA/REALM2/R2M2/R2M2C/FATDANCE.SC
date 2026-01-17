Name "Fatdance"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
SightRange			1		//  can see 
IgnoreAlertNear			TRUE
AlertGroup			TRUE
IgnoreAlertGroup			TRUE
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0

TargetPlayer					//  target the player
CanBeMoved			FALSE		//  don't allow outside forces to push the monster

///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	TurnToAngle 270
	LoopAnim	False
	StartAnim	"AmbA"
	Wait		1
	StartAnim "AttackB"
	Wait		1
	StartAnim "AmbA"
	Wait		1
	StartAnim "AttackB"
	Wait		1
	StartAnim "DeathA"	
	Wait		1
	StartAnim "Fall"
	Wait		1
	StartAnim "pose"	
	Wait		.75
	
Goto Loop1

