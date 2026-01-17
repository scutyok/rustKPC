Name "jackwalk"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			1.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		10000		//	check for enemies every 1/2 second
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

Loop1:
OnEnemyOutOfRange_Goto		DISABLE		//	don't branch on enemy out of range
						//	don't branch if the timer expires!! This can happen if the enemy goes out
						//	of range during the Attack routine and the time is still ticking!!
OnTimerExpired_Goto		DISABLE	
TimeOut				10.0


Walk						//	walk to the path3 and wait
StartAnim				"Walk"
TurnTo				tar01pos01
MoveTo				tar01pos01
StartAnim	"AmbA"
Wait				.5

//	turn to look at path2 ...
StartAnim				"Walk"
TurnTo				tar01pos02
MoveTo				tar01pos02
TurnToTarget			0.57
StartAnim				"AmbA"
Wait				.5

Goto	Loop1

