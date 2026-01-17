Name "Stump"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			1024.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster


///////////////////////////////////////////////////////////////////////////////
//	First Walk with Laughing Ambient Animation
///////////////////////////////////////////////////////////////////////////////

Walk						
StartAnim	"Run"

MoveTo		StumpPos1
StartAnim	"AmbB"
Wait		2
StartAnim	"AmbA"
Wait		2



