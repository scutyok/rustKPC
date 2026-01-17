Name "stair01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			5000		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
TargetPlayer					//	target the player
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster
Walk						//	walk to the path3 and wait

///////////////////////////////////////////////////////////////////////////////
//	Patrolling, not attacking...
///////////////////////////////////////////////////////////////////////////////

TimeOut				10.0
StartAnim				"Walk"
TurnTo				stair01pos01
MoveTo				stair01pos01
TurnToTarget			0.57		//	if the player is in range, attack
StartAnim	"AmbA"
TargetObject			clowntarget01
Wait				4
TurnTo				clowntarget01
RangedAttackTarget
StartAnim	"AmbA"
end
