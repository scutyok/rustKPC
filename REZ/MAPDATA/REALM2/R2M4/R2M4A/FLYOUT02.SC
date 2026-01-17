Name "flyout02"

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

TurnTo				fout02pos01
MoveTo				fout02pos01

TurnTo				fout02pos02
MoveTo				fout02pos02

TurnTo				fout02pos03
MoveTo				fout02pos03

TurnTo				fout02pos04
MoveTo				fout02pos04

TurnToTarget			.57




