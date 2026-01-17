Name "vict01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			512.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetObject			"dumb01"	//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

StartAnim				"Walk"
TurnTo				bp01pos01
MoveTo				bp01pos01
TurnTo				bp01pos02
MoveTo				bp01pos02
TurnTo				bp01pos03
MoveTo				bp01pos03
StartAnim				"AmbA"
Jump				600
wait				1
MeleeAttackTarget
StartAnim				"walk"
TurnTo				bp01pos04
MoveTo				bp01pos04
TargetObject			"dumb02"	
StartAnim				"AmbA"
Jump				600
wait				1
MeleeAttackTarget
wait				1
StartAnim				"AmbA"