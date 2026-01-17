Name "wake01"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			Ture		//  abort the script if we're damaged
AbortOnEnemySighted		True		//  Don't abort if we sight an enemy
SightRange			2400.0		//  can see 
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0
CheckForEnemyInterval		0.5		//  check for enemies every 1/2 second
TargetPlayer					//  target the player
CanBeMoved			FALSE		//  don't allow outside forces to push the monster



///////////////////////////////////////////////////////////////////////////////
//	If he can see the player then he shoots at the player.
///////////////////////////////////////////////////////////////////////////////
Loop1:
	OnEnemyOutOfRange_Goto		DISABLE
	OnDamage_Goto			DISABLE

	LoopAnim	TRUE
	StartAnim	"AmbA"

	Wait		999.0
	Goto		Loop1

