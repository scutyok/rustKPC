Name "Amb"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
SightRange			5000		//  can see
//OuterZone			5000 
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0
//CheckForEnemyInterval		0.5		//  check for enemies every 1/2 second
TargetPlayer					//  target the player
CanBeMoved			FALSE		//  don't allow outside forces to push the monster


///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	

	LoopAnim	TRUE
	StartAnim	"AmbA"

	wait 100000000
	Goto		Loop1

