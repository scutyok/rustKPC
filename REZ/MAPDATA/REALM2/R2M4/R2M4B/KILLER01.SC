Name "killer01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			512.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetGroup			"butpusher01"	//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

StartAnim				"Walk"
TurnTo				kr01pos01
MoveTo				kr01pos01
TurnTo				kr01pos02
MoveTo				kr01pos02
TurnTo				kr01pos03
MoveTo				kr01pos03
TurnToTarget			0.57
RangedAttackTarget
StartAnim				"AmbA"

