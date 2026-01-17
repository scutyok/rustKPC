Name "wait01"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
IgnoreAlertNear			TRUE
SightRange			1.0		//  can see 
TimeOut				999.0
//WalkSpeed			70
//RunSpeed			120
YawRate				360.0
CheckForEnemyInterval		0.5		//  check for enemies every 1/2 second
TargetPlayer					//  target the player
CanBeMoved			True		//  don't allow outside forces to push the monster

Loop1:
Wait 1000000
Goto loop1