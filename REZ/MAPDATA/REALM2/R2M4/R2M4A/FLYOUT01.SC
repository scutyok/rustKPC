Name "flyout01"

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
LoopAnim			True

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

TimeOut				10.0
Walk						//	walk to the path3 and wait
StartAnim				"Fly"

TurnTo				fout01pos01
MoveTo				fout01pos01

TurnTo				fout01pos02
MoveTo				fout01pos02

TurnTo				fout01pos03
MoveTo				fout01pos03

TurnTo				fout01pos04
MoveTo				fout01pos04

TurnToTarget			.57


