Name "cs3uni1"

///////////////////////////////////////////////////////////////////////////////
//	initial set up 
///////////////////////////////////////////////////////////////////////////////
AbortOnDamage			FALSE		//  abort the script if we're damaged
AbortOnEnemySighted		FALSE		//  Don't abort if we sight an enemy
TimeOut				999.0
WalkSpeed			300
RunSpeed			300
YawRate				120
Gravity				False



///////////////////////////////////////////////////////////////////////////////
//	Move the unipsycho up the ramp to right behind the player.
///////////////////////////////////////////////////////////////////////////////
MoveToSpot:
	LoopAnim	TRUE
	StartAnim	"Run"
	FaceMoveDir 	TRUE
	MoveTo 		unipos1
	MoveTo 		unipos2
	MoveTo 		unipos3
	//StartAnim	"PainA"
	//wait .967	
	StartAnim 	"AmbA"
	wait 1.6
	End


