Name "thrower01"

///////////////////////////////////////////////////////////////////////////////
//	set up some starting states for this script
///////////////////////////////////////////////////////////////////////////////

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
SightRange			1024.0		//	can see 1024 units
TimeOut				999.0
CheckForEnemyInterval		0.5		//	check for enemies every 1/2 second
YawRate				360.0
CanBeMoved			FALSE		//  don't allow outside forces to push the monster

///////////////////////////////////////////////////////////////////////////////
//	do the meanie, attacking stuff here
///////////////////////////////////////////////////////////////////////////////

Attack:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto		DISABLE

StartAnim				"amba"
wait .5

YawRate		540
TimeOut		2.5

if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan50_Goto	Attack_path1
if_RandomLessThan75_Goto	Attack_path3

Goto Attack

//
//	Attack 2
//
Attack_path2:
TargetObject		"target02"
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget
Goto Attack
//
//	Attack 1
//
Attack_path1:
TargetObject		"target01"
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

Goto Attack
//
//	Attack 3
//
Attack_path3:
TargetObject		"target03"
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

Goto Attack