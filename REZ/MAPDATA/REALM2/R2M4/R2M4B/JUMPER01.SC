Name "jumper01"

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
wait .9

Loop1:

TimeOut				10.0

StartAnim				"backflip"
Jump 1100
turntotarget			.1
wait .1
RangedAttackTarget
wait .3
turntotarget			.1
RangedAttackTarget
wait .5
StartAnim				"amba"
wait 2
Goto Loop1


