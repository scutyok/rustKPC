Name "autokill01"

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
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

TimeOut				10.0

Walk						//	walk to the path3 and wait
StartAnim				"Walk"
TurnTo				driverpos01
MoveTo				driverpos01
TurnTo				driverpos02
MoveTo				driverpos02
TurnTo				driverpos03
MoveTo				driverpos03

TargetObject 			cartarget01
Turnto				cartarget01
RangedAttackTarget
TargetPlayer

StartAnim				"amba"
TurntoTarget			.57
